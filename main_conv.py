import os
# import wandb
import random
import logging

import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from itertools import repeat   

from module.model import ConvE, ConvKB
from module.set_seed import set_seed
from module.argument import parse_args
from module.dataset import LinkPredDataset, TrainDataset, TestDataset


try:
    args = parse_args()
except:
    args = parse_args([])


logging.basicConfig(format='', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb.login(key = open('module/wandb_key.txt', 'r').readline())
# wandb.init(project = f'ctdkg', entity = 'soyoung')
# wandb.run.name = f'{args.dataset}-{args.model}{args.seed}-embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}'
# wandb.run.save()
# wandb.config.update(args)

criterion = nn.BCEWithLogitsLoss()

# def train_conve(model, criterion, train_dataloader_head, optimizer, scheduler, args, device):
#     model.train()
#     epoch_logs = []
#     y_multihot = torch.LongTensor(args.batch_size, args.nentity)
#     for i, (s, r, os) in enumerate(train_dataloader_head):
#         s, r = s.to(device), r.to(device)

#         optimizer.zero_grad()
#         if s.size()[0] != args.batch_size:
#             y_multihot = torch.LongTensor(s.size()[0], args.nentity)

#         y_multihot.zero_()
#         y_multihot = y_multihot.scatter_(1, os, 1)
#         y_smooth = (1 - 0.1) * y_multihot.float() + 0.1 / args.nentity
#         targets = y_smooth.to(device)

#         output = model(s, r)
#         loss = criterion(output, targets)
#         loss.backward()
#         optimizer.step()
        
#         log = {
#                 'loss': loss.item()
#         }
        
#         epoch_logs.append(log)
        
#         if i % 1000 == 0:
#             logging.info('Training the model... (%d/%d)' % (i, int(len(train_dataloader_head))))
#             logging.info(log)
        
#     scheduler.step()
#     # scheduler.step(sum([log['loss'] for log in epoch_logs])/len(epoch_logs))
    
#     return epoch_logs


def train(model, criterion, device, head_loader, tail_loader, optimizer, scheduler, args):
    model.train()
    
    epoch_logs = []
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            optimizer.zero_grad()
            
            positive_sample, negative_sample, subsampling_weight, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            
            h, r, t = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
            positive_score = model(h, r, t)
            
            if mode == 'head-batch':
                head_neg = negative_sample.view(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                negative_score = model(head_neg, true_rel, true_tail)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.view(-1)
                true_head = positive_sample[:, 0].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                negative_score = model(true_head, true_rel, tail_neg)

            score = torch.cat([positive_score, negative_score])
            label = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)]).to(device)
            loss = criterion(score, label)
            
            loss.backward()
            optimizer.step()

            log = {
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


def negative_train(model, device, head_loader, tail_loader, optimizer, scheduler, args):
    model.train()
    
    epoch_logs = []
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            optimizer.zero_grad()
            
            positive_sample, negative_sample, subsampling_weight, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            
            # if args.model == 'conve':
            #     positive_score = model.valid(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
            #     positive_score = F.logsigmoid(positive_score)
            # elif args.model == 'convkb':
            h, r, t = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
            positive_score = model(h, r, t)
            positive_score = F.logsigmoid(positive_score)

            
            if mode == 'head-batch':
                head_neg = negative_sample.view(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                # if args.model == 'conve':
                #     negative_score = model.valid(head_neg, true_rel, true_tail)
                # elif args.model == 'convkb':
                negative_score = model(head_neg, true_rel, true_tail)
                    
            elif mode == 'tail-batch':
                tail_neg = negative_sample.view(-1)
                true_head = positive_sample[:, 0].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                # if args.model == 'conve':
                #     negative_score = model.valid(true_head, true_rel, tail_neg)
                # elif args.model == 'convkb':
                negative_score = model(true_head, true_rel, tail_neg)

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
    
    return epoch_logs        


@torch.no_grad()
def evaluate(model, head_loader, tail_loader, args):
    model.eval()
    
    test_logs = defaultdict(list)
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2): 
            positive_sample, negative_sample, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample[:, 1:].to(device)
            
            # if args.model == 'conve':
            #     y_pred_pos = model.valid(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
            # elif args.model == 'convkb':
            y_pred_pos = model(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
            
            if mode == 'head-batch':
                head_neg = negative_sample.reshape(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                # if args.model == 'conve':
                #     y_pred_neg = model.valid(head_neg, true_rel, true_tail)
                # elif args.model == 'convkb':
                y_pred_neg = model(head_neg, true_rel, true_tail)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.reshape(-1)
                true_head = positive_sample[:, 0].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                # if args.model == 'conve':
                #     y_pred_neg = model.valid(true_head, true_rel, tail_neg)
                # elif args.model == 'convkb':
                y_pred_neg = model(true_head, true_rel, tail_neg)
                
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


# class ConvETrainDataset(Dataset):
#     def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, count, true_head, true_tail, entity_dict):
#         self.len = len(triples['head'])
#         self.triples = triples
#         self.nentity = nentity
#         self.nrelation = nrelation
#         self.negative_sample_size = negative_sample_size
#         self.mode = mode
#         self.count = count
#         self.true_head = true_head
#         self.true_tail = true_tail
#         self.entity_dict = entity_dict
        
#     def __len__(self):
#         return self.len
    
#     def __getitem__(self, idx):
#         head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
#         head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
#         true_tails = torch.LongTensor(self.true_tail[(head.item(),relation.item())]) + self.entity_dict[tail_type][0]
            
#         return (head + self.entity_dict[head_type][0]).item(), relation.item(), true_tails.tolist()
    
#     @staticmethod
#     def collate_fn(data):
#         max_len = max(map(lambda x: len(x[2]), data))
#         for _,_, indices in data:
#             indices.extend(repeat(indices[0], max_len - len(indices)))

#         s, o, i = zip(*data)
#         return torch.LongTensor(s), torch.LongTensor(o), torch.LongTensor(i)


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

    set_seed(args.seed)
    print(f'====================== run: {args.seed} ======================')

    if args.dataset in ['gd', 'cgd', 'cgpd', 'ctd']:
        idx = random.sample(range(len(train_triples['head'])), int(len(train_triples['head'])*args.train_frac))

        train_triples['head'] = train_triples['head'][idx]
        train_triples['tail'] = train_triples['tail'][idx]
        train_triples['relation'] = train_triples['relation'][idx]
        train_triples['head_type'] = [train_triples['head_type'][i] for i in idx]
        train_triples['tail_type'] = [train_triples['tail_type'][i] for i in idx]
    
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
        
    
    # train_dataloader_head = DataLoader(
    #     ConvETrainDataset(train_triples, nentity, nrelation, 
    #         args.negative_sample_size, 'head-batch',
    #         train_count, train_true_head, train_true_tail,
    #         entity_dict), 
    #     batch_size=args.batch_size,
    #     shuffle=True, 
    #     num_workers=args.num_workers,
    #     collate_fn=ConvETrainDataset.collate_fn
    # )
    
    # Set training configuration
    if args.model == 'conve':
        model = ConvE(
            num_nodes = nentity, num_relations = nrelation, hidden_dim = args.hidden_dim
            ).to(device)
    elif args.model == 'convkb':
        model = ConvKB(nentity = nentity, nrelation = nrelation, hidden_dim = args.hidden_dim).to(device)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    init_epoch = 1
    best_val_mrr = 0
    stopupdate = 0
    
    file_name = f'embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}_seed{args.seed}'
    if args.init_checkpoint:
        check_points = torch.load(f'{save_path}/{file_name}_epoch{args.init_checkpoint}.pt')
        init_epoch = check_points['epoch'] +1
        best_val_mrr = check_points['best_mrr']
        stopupdate = check_points['stopupdate']
        model.load_state_dict(check_points['model_state_dict'])
        optimizer.load_state_dict(check_points['optimizer_state_dict'])
        scheduler.load_state_dict(check_points['scheduler_dict'])
        
    for epoch in range(init_epoch, args.num_epoch + 1):
        print(f"=== Epoch: {epoch}")
        
        if args.negative_loss:
            train_out = negative_train(model, device, train_dataloader_head, train_dataloader_tail, optimizer, scheduler, args)
        else:
            # if args.model == 'conve':
            #     train_out = train_conve(model, criterion, train_dataloader_head, optimizer, scheduler, args, device)
            # elif args.model == 'convkb':
            train_out = train(model, criterion, device, train_dataloader_head, train_dataloader_tail, optimizer, scheduler, args)
        
        train_losses = {}
        for l in train_out[0].keys():
            train_losses[l] = sum([log[l] for log in train_out])/len(train_out)
            print(f'Train {l}: {train_losses[l]:.5f}')
        
        # if args.negative_loss:
        #     wandb.log({
        #         'Train positive sample loss': train_losses['positive_sample_loss'],
        #         'Train negative sample loss': train_losses['negative_sample_loss'],
        #         'Train loss': train_losses['loss']
        #     })
        # else:
        #     wandb.log({
        #         'Train loss': train_losses['loss']
        #     })
        
        # artifact = wandb.Artifact(
        #     f'{args.dataset}_{args.model}', 
        #     type='model',
        #     metadata=vars(args))
        # artifact.add_file(f'{save_path}/{file_name}_epoch{epoch}')
        # wandb.log_artifact(artifact)
        
        if epoch % 10 == 0:
            valid_logs = evaluate(model, valid_dataloader_head, valid_dataloader_tail, args)
            valid_metrics = {}
            for metric in valid_logs:
                valid_metrics[metric[:-5]] = torch.cat(valid_logs[metric]).mean().item()       

            
            print('----------')
            print(f"Valid MRR: {valid_metrics['mrr']:.5f}")
            print(f"Valid hits@1: {valid_metrics['hits@1']:.5f}")
            print(f"Valid hits@3': {valid_metrics['hits@3']:.5f}")
            print(f"Valid hits@10': {valid_metrics['hits@10']:.5f}")
            
            test_logs = evaluate(model, test_dataloader_head, test_dataloader_tail, args)
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
            
            if valid_metrics['mrr'] > best_val_mrr:
                best_epoch = epoch
                best_val_mrr = valid_metrics['mrr']
                best_val_result = {
                    'best_epoch': best_epoch,
                    'best_val_mrr': valid_metrics['mrr'],
                    'best_val_hit1': valid_metrics['hits@1'],
                    'best_val_hit3': valid_metrics['hits@3'],
                    'best_val_hit10': valid_metrics['hits@10'],
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
                if stopupdate > 3:
                    print(f'early stop at eopch {epoch}')
                    break
    
            check_points = {'seed': args.seed,
                            'best_epoch': best_epoch,
                            'model_state_dict': model_params,
                            'optimizer_state_dict': optim_dict,
                            'scheduler_dict': scheduler_dict}
            
            file_name = f'embdim{args.hidden_dim}_gamma{args.gamma}_lr{args.learning_rate}_advtemp{args.adversarial_temperature}_seed{args.seed}.pt'
            torch.save(check_points, f'{save_path}/{file_name}')
            
            log_save_path = f'best_val_log/{args.dataset}'
            if os.path.isdir(log_save_path):
                pass
            else:
                os.makedirs(log_save_path)
            torch.save(best_val_result, f'{log_save_path}/{args.model}_{args.seed}')
            
    # wandb.log(best_val_result)

    print('')
    for metric in best_val_result.keys():
        print(f'{metric}: {best_val_result[metric]:.5f}')
    



if __name__ == '__main__':
    main()

# wandb.run.finish()