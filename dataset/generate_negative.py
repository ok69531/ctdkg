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
        
        negative_samples = []
        while len(negative_samples) < self.negative_sample_size:
        
            if self.mode == 'head-batch':
                negative_sample = torch.randint(0, self.entity_dict[head_type], (1,))
                if negative_sample.item() not in set(self.all_true_head[(relation, tail)]):
                    negative_samples.append(negative_sample)
                else:
                    pass
        
            elif self.mode == 'tail-batch':
                negative_sample = torch.randint(0, self.entity_dict[tail_type], (1,))
                if negative_sample.item() not in set(self.all_true_tail[(head, relation)]):
                    negative_samples.append(negative_sample)
                else:
                    pass
        
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)
        
        negative_samples = torch.cat(negative_samples)
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_samples, self.mode