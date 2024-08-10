import os
import logging
import requests
import itertools

import torch
from torch.utils.data import Dataset

logging.basicConfig(format='', level=logging.INFO)


class LinkPredDataset(object):
    def __init__(self, name, root = 'dataset'):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
        '''
        
        self.name = name
        self.root = os.path.join(root, self.name)
        
        self.avail_data = ['cd', 'cg-v1', 'cg-v2', 'gd', 'cgd', 'cgpd', 'ctd']
        if self.name not in self.avail_data:
            err_msg = f'Invalid dataset name: {self.name}\n'
            err_msg += 'Available datasets are as follows:\n'
            err_msg += '\n'.join(self.name)
            raise ValueError(err_msg)
        
        self.download()
    
    def download(self):
        processed_dir = os.path.join(self.root, 'processed')
        pre_processed_file_path = os.path.join(processed_dir, self.name+'.pt')
        
        if os.path.exists(pre_processed_file_path):
            self.graph = torch.load(pre_processed_file_path)
        
        else:
            logging.info('>>> This process will be time-consuming ... ')
            logging.info('>>> Making directory ...')
            os.makedirs(processed_dir)
            
            # download full graph
            logging.info(f'>>> Downloading {self.name.upper()} graph ...')
            if 'cg' in self.name:
                ver = self.name.split('-')[1]
                data_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/cg/{ver}/{self.name}.pt?download=true'
            else:
                data_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/{self.name}/{self.name}.pt?download=true'
            response = requests.get(data_url)
            
            with open(pre_processed_file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f'    {self.name.upper()} graph is downloaded.')
            
            # download train/validation/test data
            logging.info(f'>>> Downloading splitted {self.name.upper()} graph ...')
            split_types = ['train', 'valid', 'test']
            for split_type in split_types:
                if 'cg' in self.name:
                    split_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/cg/{ver}/{split_type}_{self.name}.pt?download=true'
                else:
                    split_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/{self.name}/{split_type}_{self.name}.pt?download=true'
                response = requests.get(split_url)
                
                with open(f'{self.root}/{split_type}_{self.name}.pt', 'wb') as f:
                    f.write(response.content)
            logging.info(f'    Training/Validation/Test graphs were downloaded.')
            
            # download mapping of entyties and relations
            logging.info(f'>>> Downloading the mapping of entities and relations ...')
            map_types = ['rel_type', 'chem', 'gene', 'dis', 'pheno', 'path', 'go']
            for map_type in map_types:
                if 'cg' in self.name:
                    map_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/cg/{ver}/{map_type}_map?download=true'
                else:
                    map_url = f'https://huggingface.co/datasets/soyoungc/CTDKG/resolve/main/{self.name}/{map_type}_map?download=true'
                response = requests.get(map_url)
                
                if response.status_code == 200:
                    with open(os.path.join(processed_dir, map_type+'_map'), 'wb') as f:
                        f.write(response.content)
            logging.info(f'    Mappings were downloaded.')
            logging.info(f'>>> All elements were downloaded.')
            
            self.graph = torch.load(pre_processed_file_path)

    
    def get_edge_split(self):
        train_data = torch.load(os.path.join(self.root, 'train_'+self.name+'.pt'))
        valid_data = torch.load(os.path.join(self.root, 'valid_'+self.name+'.pt'))
        test_data = torch.load(os.path.join(self.root, 'test_'+self.name+'.pt'))

        train_triplet = make_triplet(train_data, is_train = True)
        valid_triplet = make_triplet(valid_data)
        test_triplet = make_triplet(test_data)

        return train_triplet, valid_triplet, test_triplet
    
    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph
    
    def __len__(self):
        return 1


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
        head, relation, tail = self.triples['head'][idx].item(), self.triples['relation'][idx].item(), self.triples['tail'][idx].item()
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
