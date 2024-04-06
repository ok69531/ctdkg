import os
import wandb
import logging

import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import torch_geometric
from torch_geometric.nn import GAE

from model import GNNEncoder, DistMultDecoder
from set_seed import set_seed
from argument import parse_args
from dataloader import load_data, TrainDataset, TestDataset


try:
    args = parse_args()
except:
    args = parse_args([])


logging.basicConfig(format='', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'cuda is available: {torch.cuda.is_available()}')

# wandb.login(key = open('wandb_key.txt', 'r').readline())
# wandb.init(project = f'ctdkg', entity = 'soyoung')
# wandb.run.name = f'{args.dataset}-{args.model}-embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}'
# wandb.run.save()


def train(model, device, edge_index, edge_type, head_loader, tail_loader, optimizer, scheduler, args):
    model.train()
    
    epoch_logs = []
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            optimizer.zero_grad()
            
            positive_sample, negative_sample, subsampling_weight, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            
            node_embedding = model.encode(edge_index, edge_type)
            positive_score = model.decode(node_embedding[positive_sample[:, 0]], node_embedding[positive_sample[:, 2]], positive_sample[:, 1])
            positive_score = F.logsigmoid(positive_score)
            
            if mode == 'head-batch':
                head_neg = negative_sample.view(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                negative_score = model.decode(node_embedding[head_neg], node_embedding[true_tail], true_rel)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.view(-1)
                true_head = positive_sample[:, 0].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                negative_score = model.decode(node_embedding[true_head], node_embedding[tail_neg], true_rel)

            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = negative_score.view(negative_sample.shape)
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                negative_score = F.logsigmoid(-negative_score)


            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss)/2
            loss.backward()
            optimizer.step()

            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }
            
            epoch_logs.append(log)
        
        if i % 1000 == 0:
            logging.info('Training the model... (%d/%d)' % (i, int(len(head_loader))))
            logging.info(log)
     
    scheduler.step()
    # scheduler.step(sum([log['positive_sample_loss'] for log in epoch_logs])/len(epoch_logs))
    # scheduler.step(sum([log['loss'] for log in epoch_logs])/len(epoch_logs))
    
    return epoch_logs


@torch.no_grad()
def evaluate(model, edge_index, edge_type, head_loader, tail_loader, args):
    model.eval()
    
    node_embedding = model.encode(edge_index, edge_type)
    
    test_logs = defaultdict(list)
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            positive_sample, negative_sample, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample[:, 1:].to(device)
            
            y_pred_pos = model.decode(node_embedding[positive_sample[:, 0]], node_embedding[positive_sample[:, 2]], positive_sample[:, 1])
            
            if mode == 'head-batch':
                head_neg = negative_sample.reshape(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                y_pred_neg = model.decode(node_embedding[head_neg], node_embedding[true_tail], true_rel)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.reshape(-1)
                true_head = positive_sample[:, 0].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                y_pred_neg = model.decode(node_embedding[true_head], node_embedding[tail_neg], true_rel)
            y_pred_neg = y_pred_neg.view(negative_sample.shape)

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
            logging.info('Evaluating the model... (%d/%d)' % (i, int(len(head_loader))))

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
    for i in tqdm(range(len(train_triples['head']))):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        head_type, tail_type = train_triples['head_type'][i], train_triples['tail_type'][i]
        train_count[(head, relation, head_type)] += 1
        train_count[(tail, -relation-1, tail_type)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)

    dataset_head = torch.cat((train_triples['head'], valid_triples['head'], test_triples['head']))
    dataset_tail = torch.cat((train_triples['tail'], valid_triples['tail'], test_triples['tail']))
    dataset_head_type = np.concatenate((train_triples['head_type'], valid_triples['head_type'], test_triples['head_type']))
    dataset_tail_type = np.concatenate((train_triples['tail_type'], valid_triples['tail_type'], test_triples['tail_type']))

    for i in tqdm(range(len(dataset_head))):
        dataset_head[i] = dataset_head[i] + entity_dict[dataset_head_type[i]][0]
        dataset_tail[i] = dataset_tail[i] + entity_dict[dataset_tail_type[i]][0]

    edge_index = torch.stack((dataset_head, dataset_tail)).to(device)
    edge_type = torch.cat((train_triples['relation'], valid_triples['relation'], test_triples['relation'])).to(device)
    
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
        model = GAE(
            GNNEncoder(num_nodes = nentity, num_relations = nrelation, hidden_dim = args.hidden_dim, gnn_model = args.model, num_layers = 2),
            DistMultDecoder(nrelation, args.hidden_dim)
        ).to(device)
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.learning_rate
        )
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        
        for epoch in range(1, args.num_epoch + 1):
            print(f"=== Epoch: {epoch}")
            
            train_out = train(model, device, edge_index, edge_type, train_dataloader_head, train_dataloader_tail, optimizer, scheduler, args)
            
            train_losses = {}
            for l in train_out[0].keys():
                train_losses[l] = sum([log[l] for log in train_out])/len(train_out)
                print(f'Train {l}: {train_losses[l]:.5f}')
            
            # wandb.log({
            #     'Train positive sample loss': train_losses['positive_sample_loss'],
            #     'Train negative sample loss': train_losses['negative_sample_loss'],
            #     'Train loss': train_losses['loss']
            # })
            
            if epoch % 10 == 0:
                valid_logs = evaluate(model, edge_index, edge_type, valid_dataloader_head, valid_dataloader_tail, args)
                valid_metrics = {}
                for metric in valid_logs:
                    valid_metrics[metric[:-5]] = torch.cat(valid_logs[metric]).mean().item()       

                
                print('----------')
                print(f"Valid MRR: {valid_metrics['mrr']:.5f}")
                print(f"Valid hits@1: {valid_metrics['hits@1']:.5f}")
                print(f"Valid hits@3': {valid_metrics['hits@3']:.5f}")
                print(f"Valid hits@10': {valid_metrics['hits@10']:.5f}")
                
                test_logs = evaluate(model, edge_index, edge_type, test_dataloader_head, test_dataloader_tail, args)
                test_metrics = {}
                for metric in test_logs:
                    test_metrics[metric[:-5]] = torch.cat(test_logs[metric]).mean().item()       
                
                print('----------')
                print(f"Test MRR: {test_metrics['mrr']:.5f}")
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
        
        file_name = f'embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}_seed{seed}.pt'
        torch.save(check_points, f'{save_path}/{file_name}')


if __name__ == '__main__':
    main()

# wandb.run.finish()
