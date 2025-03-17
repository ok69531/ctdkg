import os
import wandb
import random
import logging

from copy import deepcopy
from tqdm.auto import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from module.model import KGEModel
from module.set_seed import set_seed
from module.argument import parse_args
from module.dataset import LinkPredDataset, TrainDataset, TestDataset, BidirectionalOneShotIterator
from compile_neighbors import CompilERegularizer, DURA

try:
    args = parse_args()
except:
    args = parse_args([])


logging.basicConfig(format='', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'cuda is available: {torch.cuda.is_available()}')

wandb.login(key = open('module/wandb_key.txt', 'r').readline())
wandb.init(project = f'ctdkg', entity = 'soyoung')
wandb.run.name = f'{args.dataset}-{args.model}{args.seed}'
# wandb.run.name = f'{args.dataset}-{args.model}{args.seed}-embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}'
wandb.run.save()
wandb.config.update(args)


def train(model, device, train_iterator, optimizer, scheduler, args):
    model.train()
    
    optimizer.zero_grad()
    
    positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
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

    # if args.model.upper() == 'HOUSE':
    #     if mode == 'head-batch':
    #         pos_part = positive_sample[:, 0].unsqueeze(dim=1)
    #     else:
    #         pos_part = positive_sample[:, 2].unsqueeze(dim=1)
    #     positive_score = model((positive_sample, pos_part), mode=mode)
    # else:
    
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
    
    scheduler.step()
    
    return log


# def train(model, device, head_loader, tail_loader, optimizer, scheduler, args):
#     model.train()
    
#     epoch_logs = []
#     for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
#         for b in (b1, b2):
#             optimizer.zero_grad()
            
#             positive_sample, negative_sample, subsampling_weight, mode = b
#             positive_sample = positive_sample.to(device)
#             negative_sample = negative_sample.to(device)
#             subsampling_weight = subsampling_weight.to(device)
            
#             negative_score = model((positive_sample, negative_sample), mode=mode)
#             if args.negative_adversarial_sampling:
#                 #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
#                 negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
#                                 * F.logsigmoid(-negative_score)).sum(dim = 1)
#             else:
#                 negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

#             if args.model.upper() == 'HOUSE':
#                 if mode == 'head-batch':
#                     pos_part = positive_sample[:, 0].unsqueeze(dim=1)
#                 else:
#                     pos_part = positive_sample[:, 2].unsqueeze(dim=1)
#                 positive_score = model((positive_sample, pos_part), mode=mode)
#             else:
#                 positive_score = model(positive_sample)
#             positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

#             if args.uni_weight:
#                 positive_sample_loss = - positive_score.mean()
#                 negative_sample_loss = - negative_score.mean()
#             else:
#                 positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
#                 negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

#             loss = (positive_sample_loss + negative_sample_loss)/2
            
#             if args.regularization != 0.0:
#                 #Use L3 regularization for ComplEx and DistMult
#                 regularization = args.regularization * (
#                     model.entity_embedding.norm(p = 3)**3 + 
#                     model.relation_embedding.norm(p = 3).norm(p = 3)**3
#                 )
#                 loss = loss + regularization
#                 regularization_log = {'regularization': regularization.item()}
#             else:
#                 regularization_log = {}
            
#             loss.backward()

#             optimizer.step()

#             log = {
#                 **regularization_log,
#                 'positive_sample_loss': positive_sample_loss.item(),
#                 'negative_sample_loss': negative_sample_loss.item(),
#                 'loss': loss.item()
#             }
            
#             epoch_logs.append(log)
        
#         if i % 1000 == 0:
#             logging.info('Training the model... (%d/%d)' % (i, int(len(head_loader))))
#             logging.info(log)
     
#     scheduler.step()
#     # scheduler.step(sum([log['positive_sample_loss'] for log in epoch_logs])/len(epoch_logs))
#     # scheduler.step(sum([log['loss'] for log in epoch_logs])/len(epoch_logs))
    
#     return epoch_logs


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
            mr_list = ranking_list.to(torch.float)
            
            batch_results = {'hits@1_list': hits1_list,
                            'hits@3_list': hits3_list,
                            'hits@10_list': hits10_list,
                            'mrr_list': mrr_list,
                            'mr_list': mr_list}
            
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
    
    dataset = LinkPredDataset(args.dataset)
    data = dataset[0]
    train_triples, valid_triples, test_triples = dataset.get_edge_split()
    
    nrelation = data.num_relations
    entity_dict = dict()
    cur_idx = 0
    for key in data['num_nodes_dict']:
        entity_dict[key] = (cur_idx, cur_idx + data['num_nodes_dict'][key])
        cur_idx += data['num_nodes_dict'][key]
    nentity = sum(data['num_nodes_dict'].values())

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
        head, relation, tail = train_triples['head'][i].item(), train_triples['relation'][i].item(), train_triples['tail'][i].item()
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


    # for seed in range(args.num_runs):
    set_seed(args.seed)
    print(f'====================== run: {args.seed} ======================')

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
    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
    
    # Set training configuration
    # if args.model.upper() == 'GIE':
    #     model = GIE(
    #         nentity=nentity,
    #         nrelation=nrelation,
    #         hidden_dim=args.hidden_dim,
    #         gamma=args.gamma,
    #         bias=args.bias, init_size=args.init_size
    #     ).to(device)
    # elif args.model.upper() == 'HOUSE':
    #     model = HousE(
    #         nentity=nentity,
    #         nrelation=nrelation,
    #         hidden_dim=args.hidden_dim,
    #         gamma=args.gamma,
    #         house_dim=args.house_dim, housd_num=args.housd_num, thred=args.thred
    #     ).to(device)
    # else:
    
    model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        num_entity_embedding=args.num_entity_embedding,
        num_relation_embedding=args.num_relation_embedding
    ).to(device)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )
    scheduler = StepLR(optimizer, step_size=30000, gamma=0.8)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    init_epoch = 1
    best_val_mrr = 0
    stopupdate = 0
    
    if args.init_checkpoint:
        check_points = torch.load(f'{save_path}/{file_name}_epoch{args.init_checkpoint}.pt')
        init_epoch = check_points['epoch'] +1
        best_val_mrr = check_points['best_mrr']
        stopupdate = check_points['stopupdate']
        model.load_state_dict(check_points['model_state_dict'])
        optimizer.load_state_dict(check_points['optimizer_state_dict'])
        scheduler.load_state_dict(check_points['scheduler_dict'])
    
    for i in range(1, args.max_step + 1):
        train_out = train(model, device, train_iterator, optimizer, scheduler, args)
        
        if i % 100 == 0:
            print(f"=== {i}-th iteration")
            logging.info('Training the model... (%d/%d)' % (i, args.max_step))
            for log in train_out.keys():
                logging.info(f'Train {log}: {train_out[log]:.5f}')
        
        wandb.log({
            'Train positive sample loss': train_out['positive_sample_loss'],
            'Train negative sample loss': train_out['negative_sample_loss'],
            'Train loss': train_out['loss']
        })
        
        # check_points = {
        #     'seed': args.seed,
        #     'iteration': i,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_dict': scheduler.state_dict(),
        #     'best_mrr': best_val_mrr,
        #     'stopupdate': stopupdate}

        # artifact = wandb.Artifact(
        #     f'{args.dataset}_{args.model}', 
        #     type='model',
        #     metadata=vars(args))
        # artifact.add_file(f'{save_path}/{file_name}_epoch{epoch}.pt')
        # wandb.log_artifact(artifact)
        
        if i % 1000 == 0:
            valid_logs = evaluate(model, valid_dataloader_head, valid_dataloader_tail, args)
            valid_metrics = {}
            for metric in valid_logs:
                valid_metrics[metric[:-5]] = torch.cat(valid_logs[metric]).mean().item()       
            
            print('----------')
            print(f"Valid MR: {valid_metrics['mr']:.5f}")
            print(f"Valid MRR: {valid_metrics['mrr']:.5f}")
            print(f"Valid hits@1: {valid_metrics['hits@1']:.5f}")
            print(f"Valid hits@3: {valid_metrics['hits@3']:.5f}")
            print(f"Valid hits@10: {valid_metrics['hits@10']:.5f}")
            
            test_logs = evaluate(model, test_dataloader_head, test_dataloader_tail, args)
            test_metrics = {}
            for metric in test_logs:
                test_metrics[metric[:-5]] = torch.cat(test_logs[metric]).mean().item()       
            
            print('----------')
            print(f"Test MR: {test_metrics['mr']:.5f}")
            print(f"Test MRR: {test_metrics['mrr']:.5f}")
            print(f"Test hits@1: {test_metrics['hits@1']:.5f}")
            print(f"Test hits@3: {test_metrics['hits@3']:.5f}")
            print(f"Test hits@10': {test_metrics['hits@10']:.5f}")
            
            wandb.log({
                'Valid MRR': valid_metrics['mrr'],
                'Valid hits@1': valid_metrics['hits@1'],
                'Valid hits@3': valid_metrics['hits@3'],
                'Valid hits@10': valid_metrics['hits@10'],
                'Test MRR': test_metrics['mrr'],
                'Test hits@1': test_metrics['hits@1'],
                'Test hits@3': test_metrics['hits@3'],
                'Test hits@10': test_metrics['hits@10']
            })
            
            if valid_metrics['mrr'] > best_val_mrr:
                best_iters = i
                best_val_mrr = valid_metrics['mrr']
                best_val_result = {
                    'best_iters': best_iters,
                    'best_val_mr': valid_metrics['mr'],
                    'best_val_mrr': valid_metrics['mrr'],
                    'best_val_hit1': valid_metrics['hits@1'],
                    'best_val_hit3': valid_metrics['hits@3'],
                    'best_val_hit10': valid_metrics['hits@10'],
                    'final_test_mr': test_metrics['mr'],
                    'final_test_mrr': test_metrics['mrr'],
                    'final_test_hit1': test_metrics['hits@1'],
                    'final_test_hit3': test_metrics['hits@3'],
                    'final_test_hit10': test_metrics['hits@10']
                }
                
                model_params = deepcopy(model.state_dict())
                optim_dict = deepcopy(optimizer.state_dict())
                scheduler_dict = deepcopy(scheduler.state_dict())
                stopupdate = 0
                
            else:
                stopupdate += 1
        
        if stopupdate > 5:
            print(f'early stop at iteration {i}')
            break
    
    wandb.log(best_val_result)
    # saving checkpoints and results
    check_points = {'seed': args.seed,
                    'best_iters': best_iters,
                    'model_state_dict': model_params,
                    'optimizer_state_dict': optim_dict,
                    'scheduler_dict': scheduler_dict}
    
    file_name = f'seed{args.seed}.pt'
    # file_name = f'embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}_seed{args.seed}.pt'
    torch.save(check_points, os.path.join(save_path, file_name))

    log_save_path = f'best_val_log/{args.dataset}'
    if os.path.isdir(log_save_path):
        pass
    else:
        os.makedirs(log_save_path)
    torch.save(best_val_result, f'{log_save_path}/{args.model}_{args.seed}')
        
    print('')
    for metric in best_val_result.keys():
        print(f'{metric}: {best_val_result[metric]:.5f}')
    

if __name__ == '__main__':
    main()

wandb.run.finish()
