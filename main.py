import os
import wandb

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import KGEModel
from set_seed import set_seed
from argument import parse_args
from dataloader import load_data, TrainDataset, TestDataset


try:
    args = parse_args()
except:
    args = parse_args([])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb.login(key = open('wandb_key.txt', 'r').readline())
# wandb.init(project = f'ctdkg', entity = 'soyoung')
# wandb.run.name = f'{args.dataset}-{args.model}-embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}'
# wandb.run.save()


def train(model, device, head_loader, tail_loader, optimizer, scheduler, args):
    model.train()
    
    epoch_logs = []
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            positive_sample, negative_sample, subsampling_weight, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            
            negative_score = model((positive_sample, negative_sample), mode=mode)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

            positive_score = model(positive_sample)
            positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss)/2
            
            if args.regularization != 0.0:
                #Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 + 
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}
            
            loss.backward()

            optimizer.step()

            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }
            
            epoch_logs.append(log)
        
        if i % 100 == 0:
            print('Training the model... (%d/%d)' % (i, int(len(head_loader))))
            print(log)
     
    scheduler.step(sum([log['loss'] for log in epoch_logs])/len(epoch_logs))
    
    return epoch_logs


@torch.no_grad()
def evaluate(model, head_loader, tail_loader, args):
    model.eval()
    
    test_logs = defaultdict(list)
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            positive_sample, negative_sample, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            
            score = model((positive_sample, negative_sample), mode)

            y_pred_pos = score[:, 0]
            y_pred_neg = score[:, 1:]

            y_pred_pos = y_pred_pos.view(-1, 1)
            # optimistic rank: "how many negatives have a larger score than the positive?"
            # ~> the positive is ranked first among those with equal score
            optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
            # pessimistic rank: "how many negatives have at least the positive score?"
            # ~> the positive is ranked last among those with equal score
            pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
            ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1./ranking_list.to(torch.float)
            
            batch_results = {'hits@1_list': hits1_list,
                            'hits@3_list': hits3_list,
                            'hits@10_list': hits10_list,
                            'mrr_list': mrr_list}
            
            for metric in batch_results:
                test_logs[metric].append(batch_results[metric])

        if i % args.test_log_steps == 0:
            print('Evaluating the model... (%d/%d)' % (i, int(len(head_loader))))

    return test_logs


def main():
    save_path = f'saved_model/{args.dataset}/{args.model}'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    train_data, train_triples, valid_triples, test_triples = load_data(data_type = args.dataset)
    
    nrelation = train_data.num_relations
    entity_dict = dict()
    cur_idx = 0
    for key in train_data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + train_data['num_nodes_dict'][key])
        cur_idx += train_data['num_nodes_dict'][key]
    nentity = sum(train_data['num_nodes_dict'].values())

    args.nentity = nentity
    args.nrelation = nrelation

    print('Model: %s' % args.model)
    print('Dataset: %s' % args.dataset)
    print('#entity: %d' % nentity)
    print('#relation: %d' % nrelation)

    print('#train: %d' % len(train_triples['head']))
    print('#valid: %d' % len(valid_triples['head']))
    print('#test: %d' % len(test_triples['head']))

    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in range(len(train_triples['head'])):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
        train_count[(head, relation, head_type)] += 1
        train_count[(tail, -relation-1, tail_type)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)


    random_sampling = False
    # validation loader
    valid_dataloader_head = DataLoader(
        TestDataset(
            valid_triples, 
            args, 
            'head-batch',
            random_sampling,
            entity_dict
        ),
        batch_size = args.test_batch_size,
        num_workers = args.num_workers,
        collate_fn = TestDataset.collate_fn
    )
    valid_dataloader_tail = DataLoader(
        TestDataset(
            valid_triples, 
            args, 
            'tail-batch',
            random_sampling,
            entity_dict
        ),
        batch_size = args.test_batch_size,
        num_workers = args.num_workers,
        collate_fn = TestDataset.collate_fn
    )
    
    # test loader
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples, 
            args, 
            'head-batch',
            random_sampling,
            entity_dict
        ),
        batch_size = args.test_batch_size,
        num_workers = args.num_workers,
        collate_fn = TestDataset.collate_fn
    )
    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples, 
            args, 
            'tail-batch',
            random_sampling,
            entity_dict
        ),
        batch_size = args.test_batch_size,
        num_workers = args.num_workers,
        collate_fn = TestDataset.collate_fn
    )


    for seed in range(args.num_runs):
        set_seed(seed)
        print(f'====================== run: {seed} ======================')

        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, 
                args.negative_sample_size, 'head-batch',
                train_count, train_true_head, train_true_tail,
                entity_dict), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, 
                args.negative_sample_size, 'tail-batch',
                train_count, train_true_head, train_true_tail,
                entity_dict), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=TrainDataset.collate_fn
        )
        
        # Set training configuration
        model = KGEModel(
            model_name=args.model,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding
        ).to(device)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.learning_rate
        )
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        
        for epoch in range(1, args.num_epoch + 1):
            train_out = train(model, device, train_dataloader_head, train_dataloader_tail, optimizer, scheduler, args)
            
            train_losses = {}
            for l in train_out[0].keys():
                train_losses[l] = sum([log[l] for log in train_out])/len(train_out)
                print(f'Traain {l} at epoch {epoch}: {train_losses[l]}')
            
            print(f"Train positive sample loss: {train_losses['positive_sample_loss']:.5f}")
            print(f"Train negative sample loss: {train_losses['negative_sample_loss']:.5f}")
            print(f"Train loss: {train_losses['loss']:.5f}")
            
            # wandb.log({
            #     'Train positive sample loss': train_losses['positive_sample_loss'],
            #     'Train negative sample loss': train_losses['negative_sample_loss'],
            #     'Train loss': train_losses['loss']
            # })
            
            if epoch % 10:
                valid_logs = evaluate(model, valid_dataloader_head, valid_dataloader_tail, args)
                valid_metrics = {}
                for metric in valid_logs:
                    valid_metrics[metric[:-5]] = torch.cat(valid_logs[metric]).mean().item()       

                print(f"=== Epoch: {epoch}")
                print(f"Valid MRR: {valid_metrics['mrr']:.5f}")
                print(f"Valid hits@1: {valid_metrics['hits@1']:.5f}")
                print(f"Valid hits@3': {valid_metrics['hits@3']:.5f}")
                print(f"Valid hits@10': {valid_metrics['hits@10']:.5f}")
                
                test_logs = evaluate(model, test_dataloader_head, test_dataloader_tail, args)
                test_metrics = {}
                for metric in test_logs:
                    test_metrics[metric[:-5]] = torch.cat(test_logs[metric]).mean().item()       
                
                print(f"Test MRR': {test_metrics['mrr']:.5f}")
                print(f"Test hits@1': {test_metrics['hits@1']:.5f}")
                print(f"Test hits@3': {test_metrics['hits@3']:.5f}")
                print(f"Test hits@10': {test_metrics['hits@10']:.5f}")
                
                # wandb.log({
                #     'Valid MRR': valid_metrics['mrr'],
                #     'Valid hits@1': valid_metrics['hits@1'],
                #     'Valid hits@3': valid_metrics['hits@3'],
                #     'Valid hits@10': valid_metrics['hits@10'],
                #     'Test MRR': test_metrics['mrr'],
                #     'Test hits@1': test_metrics['hits@1'],
                #     'Test hits@3': test_metrics['hits@3'],
                #     'Test hits@10': test_metrics['hits@10']
                # })

        check_points = {'seed': seed,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_dict': scheduler.state_dict()}
        
        file_name = f'embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}_seed{seed}.json'
        torch.save(check_points, save_path + file_name)


if __name__ == '__main__':
    main()

# wandb.run.finish()