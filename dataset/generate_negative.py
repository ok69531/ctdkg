import torch
from torch.utils.data import Dataset


class NegativeSampling(Dataset):
    def __init__(self, triples, negative_sample_size, mode, all_true_head, all_true_tail, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.all_true_head = all_true_head
        self.all_true_tail = all_true_tail
        self.entity_dict = entity_dict
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx].item(), self.triples['relation'][idx].item(), self.triples['tail'][idx].item()
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = [head, relation, tail]
        
        if self.mode == 'head-batch':
            negative_sample = torch.randint(0, self.entity_dict[head_type], (self.negative_sample_size+100,))
            negative_sample = torch.stack([i for i in negative_sample if i.item() not in set(self.all_true_head[(relation, tail)])])[:self.negative_sample_size]
        elif self.mode == 'tail-batch':
            negative_sample = torch.randint(0, self.entity_dict[tail_type], (self.negative_sample_size+100,))
            negative_sample = torch.stack([i for i in negative_sample if i.item() not in set(self.all_true_tail[(head, relation)])])[:self.negative_sample_size]
        else:
            raise
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, self.mode
