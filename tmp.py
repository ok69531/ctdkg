#%%
import os
import logging

from tqdm.auto import tqdm
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from module.model import KGEModel
from module.set_seed import set_seed
from module.argument import parse_args
from ctdkg.module.dataset import load_data, TrainDataset, TestDataset


try:
    args = parse_args()
except:
    args = parse_args([])

args.num_workers = 0

logging.basicConfig(format='', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


seed = 42
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


#%%
import numpy as np

import torch
from torch import nn

for i, (b1, b2) in enumerate(zip(train_dataloader_head, train_dataloader_tail)):
    break
for b in (b1, b2):
    break
positive_sample, negative_sample, subsampling_weight, mode = b


# dataset_head = torch.cat((train_triples['head'], valid_triples['head'], test_triples['head']))
# dataset_tail = torch.cat((train_triples['tail'], valid_triples['tail'], test_triples['tail']))
# dataset_head_type = np.concatenate((train_triples['head_type'], valid_triples['head_type'], test_triples['head_type']))
# dataset_tail_type = np.concatenate((train_triples['tail_type'], valid_triples['tail_type'], test_triples['tail_type']))

# for i in tqdm(range(len(dataset_head))):
#     dataset_head[i] = dataset_head[i] + entity_dict[dataset_head_type[i]][0]
#     dataset_tail[i] = dataset_tail[i] + entity_dict[dataset_tail_type[i]][0]

# edge_index = torch.stack((dataset_head, dataset_tail)).to(device)
# edge_type = torch.cat((train_triples['relation'], valid_triples['relation'], test_triples['relation'])).to(device)

# edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
# edge_type = torch.cat([edge_type, edge_type + nrelation])


#%%
class ConvKB(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, out_channels=64, kernel_size=1, dropout=0.3):
        super(ConvKB, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        
        self.ent_embedding = nn.Embedding(nentity, self.hidden_dim)
        self.rel_embedding = nn.Embedding(nrelation, self.hidden_dim)
        
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.out_channels, (self.kernel_size, 3))
        self.conv2_bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.hidden_dim - self.kernel_size + 1) * self.out_channels, 1, bias = False)
        
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)
    
    def _calc(self, h, r, t):
        h = h.unsqueeze(1)
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)
        
        conv_input = torch.cat([h, r, t], 1)
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_dim - self.kernel_size + 1) * self.out_channels)
        
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)
        
        return -score
    
    def forward(self, h, r, t):
        h = self.ent_embedding(h)
        r = self.rel_embedding(r)
        t = self.ent_embedding(t)
        
        score = self._calc(h, r, t)
        
        return score


#%%
args.hidden_dim = 5
model = ConvKB(nentity, nrelation, args.hidden_dim)

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


criterion = nn.BCEWithLogitsLoss()
score = torch.cat([positive_score, negative_score])
label = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)])
loss = criterion(score, label)


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
            
            if args.model == 'rgcn':
                node_embedding = model.encode(edge_index, edge_type)
                positive_score = model.decode(node_embedding[positive_sample[:, 0]], node_embedding[positive_sample[:, 2]], positive_sample[:, 1])
                positive_score = F.logsigmoid(positive_score)
            elif args.model == 'compgcn':
                h, r, t = model(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
                positive_score = h * r * t
                positive_score = torch.sum(positive_score, dim = 1)
                positive_score = F.logsigmoid(positive_score)

            
            if mode == 'head-batch':
                head_neg = negative_sample.view(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                if args.model == 'rgcn':
                    negative_score = model.decode(node_embedding[head_neg], node_embedding[true_tail], true_rel)
                elif args.model == 'compgcn':
                    h, r, t = model(head_neg, true_rel, true_tail)
                    negative_score = h * r * t
                    negative_score = torch.sum(negative_score, dim = 1)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.view(-1)
                true_head = positive_sample[:, 0].repeat_interleave(args.negative_sample_size)
                true_rel = positive_sample[:, 1].repeat_interleave(args.negative_sample_size)
                if args.model == 'rgcn':
                    negative_score = model.decode(node_embedding[true_head], node_embedding[tail_neg], true_rel)
                elif args.model == 'compgcn':
                    h, r, t = model(true_head, true_rel, tail_neg)
                    negative_score = h * r * t
                    negative_score = torch.sum(negative_score, dim = 1)

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
     
    scheduler.step(sum([log['positive_sample_loss'] for log in epoch_logs])/len(epoch_logs))
    # scheduler.step(sum([log['loss'] for log in epoch_logs])/len(epoch_logs))
    
    return epoch_logs


for i, (b1, b2) in enumerate(zip(valid_dataloader_head, valid_dataloader_tail)):
    break
for b in (b1, b2):
    break


@torch.no_grad()
def evaluate(model, edge_index, edge_type, head_loader, tail_loader, args):
    model.eval()
    
    if args.model == 'rgcn':
        node_embedding = model.encode(edge_index, edge_type)
        
    test_logs = defaultdict(list)
    for i, (b1, b2) in enumerate(zip(head_loader, tail_loader)):
        for b in (b1, b2):
            positive_sample, negative_sample, mode = b
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample[:, 1:].to(device)
            
            if args.model == 'rgcn':
                y_pred_pos = model.decode(node_embedding[positive_sample[:, 0]], node_embedding[positive_sample[:, 2]], positive_sample[:, 1])
            elif args.model == 'compgcn':
                h, r, t = model(positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2])
                y_pred_pos = h * r * t
                y_pred_pos = torch.sum(y_pred_pos, dim = 1)
            
            if mode == 'head-batch':
                head_neg = negative_sample.reshape(-1)
                true_tail = positive_sample[:, 2].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                if args.model == 'rgcn':
                    y_pred_neg = model.decode(node_embedding[head_neg], node_embedding[true_tail], true_rel)
                elif args.model == 'compgcn':
                    h, r, t = model(head_neg, true_rel, true_tail)
                    y_pred_neg = h * r * t
                    y_pred_neg = torch.sum(y_pred_neg, dim = 1)
            elif mode == 'tail-batch':
                tail_neg = negative_sample.reshape(-1)
                true_head = positive_sample[:, 0].repeat_interleave(negative_sample.size(1))
                true_rel = positive_sample[:, 1].repeat_interleave(negative_sample.size(1))
                if args.model == 'rgcn':
                    y_pred_neg = model.decode(node_embedding[true_head], node_embedding[tail_neg], true_rel)
                elif args.model == 'compgcn':
                    h, r, t = model(true_head, true_rel, tail_neg)
                    y_pred_neg = h * r * t
                    y_pred_neg = torch.sum(y_pred_neg, dim = 1)
                    
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