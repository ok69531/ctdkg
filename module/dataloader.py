from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
from torch.utils.data import Dataset


def make_triplet(data, is_train = False):
    head_type = []
    tail_type = []
    relation = []
    head = []
    tail = []
    if not is_train:
        head_neg = []
        tail_neg = []

    for (h, r, t), e in data.edge_index_dict.items():
        head_type.append(list(itertools.repeat(h, e.shape[1])))
        tail_type.append(list(itertools.repeat(t, e.shape[1])))
        relation.append(data.edge_reltype[(h, r, t)].view(-1))
        head.append(e[0])
        tail.append(e[1])
        if not is_train:
            head_neg.append(data.head_neg[(h, r, t)])
            tail_neg.append(data.tail_neg[(h, r, t)])

    try:
        triples = {
            'head_type': list(itertools.chain(*head_type)),
            'head': torch.cat(head),
            'head_neg': torch.cat(head_neg),
            'relation': torch.cat(relation),
            'tail_type': list(itertools.chain(*tail_type)),
            'tail': torch.cat(tail),
            'tail_neg': torch.cat(tail_neg)
        }
    except:
        triples = {
            'head_type': list(itertools.chain(*head_type)),
            'head': torch.cat(head),
            'relation': torch.cat(relation),
            'tail_type': list(itertools.chain(*tail_type)),
            'tail': torch.cat(tail)
        }
    
    return triples


def load_data(data_type):
    if 'cg' in data_type:
        data_type, ver = data_type.split('-')
        train_path = f'dataset/processed/{data_type}/{ver}/train_{data_type}-{ver}.pt'
        valid_path = f'dataset/processed/{data_type}/{ver}/valid_{data_type}-{ver}.pt'
        test_path = f'dataset/processed/{data_type}/{ver}/test_{data_type}-{ver}.pt'
    else:
        train_path = f'dataset/processed/{data_type}/train_{data_type}.pt'
        valid_path = f'dataset/processed/{data_type}/valid_{data_type}.pt'
        test_path = f'dataset/processed/{data_type}/test_{data_type}.pt'
                
    train_data = torch.load(train_path)
    valid_data = torch.load(valid_path)
    test_data = torch.load(test_path)
    
    train_triplet = make_triplet(train_data, is_train = True)
    valid_triplet = make_triplet(valid_data)
    test_triplet = make_triplet(test_data)
    
    return train_data, train_triplet, valid_triplet, test_triplet


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, count, true_head, true_tail, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = count
        self.true_head = true_head
        self.true_tail = true_tail
        self.entity_dict = entity_dict
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = [head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]]

        subsampling_weight = self.count[(head, relation, head_type)] + self.count[(tail, -relation-1, tail_type)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if self.mode == 'head-batch':
            negative_sample = torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1], (self.negative_sample_size,))
        elif self.mode == 'tail-batch':
            negative_sample = torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], (self.negative_sample_size,))
        else:
            raise
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train
        self.entity_dict = entity_dict

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = torch.LongTensor((head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]), 
                                             self.triples['head_neg'][idx] + self.entity_dict[head_type][0]])
            else:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]), 
                        torch.randint(self.entity_dict[head_type][0], self.entity_dict[head_type][1], size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]), 
                                             self.triples['tail_neg'][idx] + self.entity_dict[tail_type][0]])
            else:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]), 
                        torch.randint(self.entity_dict[tail_type][0], self.entity_dict[tail_type][1], size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode
