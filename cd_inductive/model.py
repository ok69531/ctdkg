# https://github.com/snap-stanford/ogb/tree/master
# RoataE: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
# ConvE: https://github.com/TimDettmers/ConvE
# ConvKB: https://github.com/daiquocnguyen/ConvKB
# CompGCN: https://github.com/malllabiisc/CompGCN

import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from torch_geometric.nn import RGCNConv

from module.utils.euclidean import givens_rotations
from module.utils.hyperbolic import mobius_add, expmap0, project, expmap1, logmap1, hyp_distance_multi_c
# from module.compgcn_layer import CompGCNConv, CompGCNConvBasis


class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        
        self.dataset = args.dataset
        self.model_name = args.model
        
        self.llm_model = args.llm_model
        self.embedding_type = args.embedding_type
        
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.hidden_dim = args.hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]), 
            requires_grad=False
        )
        
        self.entity_dim = args.hidden_dim * args.num_entity_embedding
        self.relation_dim = args.hidden_dim * args.num_relation_embedding
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / args.hidden_dim]), 
            requires_grad=False
        )

        if (args.embedding_type == 'vanilla') or (args.embedding_type == 'concat'):
            self.entity_id_embedding = nn.Parameter(torch.zeros(args.nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_id_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            
            self.relation_embedding = nn.Parameter(torch.zeros(args.nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
            if args.model == 'HAKE':
                nn.init.ones_(
                    tensor=self.relation_embedding[:, args.hidden_dim : 2*args.hidden_dim]
                )
                nn.init.zeros_(
                    tensor=self.relation_embedding[:, 2*args.hidden_dim : 3*args.hidden_dim]
                )
                
            if args.model == 'Rotate4D':
                nn.init.ones_(tensor=self.relation_embedding[:, 3*args.hidden_dim:4*args.hidden_dim])
        
        if args.model == 'HAKE':
            self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
            self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]))
        
        if args.model == 'QuatRE':
            if args.embedding_type == 'concat':
                self.Whr = nn.Parameter(torch.zeros(args.nrelation, 4 * (args.hidden_dim * 2)))
                self.Wtr = nn.Parameter(torch.zeros(args.nrelation, 4 * (args.hidden_dim * 2)))
                nn.init.xavier_uniform_(self.Whr)
                nn.init.xavier_uniform_(self.Wtr)
            else:
                self.Whr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
                self.Wtr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
                nn.init.xavier_uniform_(self.Whr)
                nn.init.xavier_uniform_(self.Wtr)
        
        if (args.embedding_type == 'text') or (args.embedding_type == 'concat'):
            self.load_embedding()
            
            self.entity_mlp = nn.Linear(self.entity_desc_embedding.size(1), self.entity_dim)
            self.relation_mlp = nn.Linear(self.rel_desc_embedding.size(1), self.relation_dim)
            self.mol_mlp = nn.Linear(self.molecule_embedding.size(1), self.entity_dim)
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if args.model not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'HAKE', 'TripleRE', 'QuatRE', 'Rotate4D']:
            raise ValueError('model %s not supported' % args.model)
            
        if args.model == 'RotatE' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 1):
            raise ValueError('RotatE should use --num_entity_embedding 2')

        if args.model == 'ComplEx' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 2):
            raise ValueError('ComplEx should use --num_entity_embedding 2 and --num_relation_embedding 2')
        
        if args.model == 'HAKE' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 3):
            raise ValueError('HAKE should use --num_entity_embedding 2 and --num_relation_embedding 3')
        
        if args.model == 'TripleRE' and (args.num_relation_embedding != 3):
            raise ValueError('TripleRE should use --num_relation_embedding 3')
        
        if (args.model == 'QuatRE' or args.model == 'Rotate4D') and (args.num_entity_embedding != 4 or args.num_relation_embedding != 4):
            raise ValueError('QuatRE or Rotate4D should use --num_entity_embedding 4 and --num_relation_embedding 4')

    def load_embedding(self):
        self.entity_desc_embedding = torch.load('dataset/cd/processed/biot5+_entity_embedding')
        self.rel_desc_embedding = torch.load('dataset/cd/processed/biot5+_relation_embedding')
        self.molecule_embedding = torch.load('dataset/cd/processed/molecule_embedding')
        
        self.test_desc_embedding = torch.load('dataset/cd/processed/test_biot5+_chemical_embedding')
        self.test_molecule_embedding = torch.load('dataset/cd/processed/test_molecule_embedding')
        
        # self.entity_desc_embedding  = self.entity_desc_embedding.to(torch.float32)
        # self.rel_desc_embedding  = self.rel_desc_embedding.to(torch.float32)
        # self.molecule_embedding = self.molecule_embedding.to(torch.float32)
        # self.test_desc_embedding = self.test_desc_embedding.to(torch.float32)
        # self.test_molecule_embedding = self.test_molecule_embedding.to(torch.float32)
        
        self.entity_desc_embedding  = self.entity_desc_embedding.to(torch.float32).cuda()
        self.rel_desc_embedding  = self.rel_desc_embedding.to(torch.float32).cuda()
        self.molecule_embedding  = self.molecule_embedding.to(torch.float32).cuda()
        self.test_desc_embedding = self.test_desc_embedding.to(torch.float32).cuda()
        self.test_molecule_embedding = self.test_molecule_embedding.to(torch.float32).cuda()
    
    def forward(self, sample, mode='single', train_type = 'train'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        if train_type != 'test':
            if self.embedding_type == 'text':
                self.entity_embedding = self.entity_mlp(self.entity_desc_embedding)
                self.relation_embedding = self.relation_mlp(self.rel_desc_embedding)
            elif self.embedding_type == 'concat':
                self.text_entity_embedding = self.entity_mlp(self.entity_desc_embedding)
                self.text_relation_embedding = self.relation_mlp(self.rel_desc_embedding)
                mol_entity_embedding = self.mol_mlp(self.molecule_embedding)
                self.entity_embedding = torch.cat([mol_entity_embedding, self.entity_id_embedding], dim = 0)
        else:
            self.text_entity_embedding = self.entity_mlp(self.entity_desc_embedding)
            test_entity_embedding = self.entity_mlp(self.test_desc_embedding)
            self.text_entity_embedding[:len(test_entity_embedding)] = test_entity_embedding
            test_mol_entity_embedding = self.mol_mlp(self.test_molecule_embedding)
            self.entity_embedding[:len(test_mol_entity_embedding)] = test_mol_entity_embedding
        
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,2]).unsqueeze(1)
            
            if self.model_name == 'QuatRE':
                self.hr = torch.index_select(self.Whr, dim=0, index=sample[:,1]).unsqueeze(1)
                self.tr = torch.index_select(self.Wtr, dim=0, index=sample[:,1]).unsqueeze(1)
            
            if self.embedding_type == 'concat':
                text_head = torch.index_select(self.text_entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
                text_relation = torch.index_select(self.text_relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
                text_tail = torch.index_select(self.text_entity_embedding, dim=0, index=sample[:,2]).unsqueeze(1)
            else:
                text_head = text_relation = text_tail = None
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            
            if self.model_name == 'QuatRE':
                self.hr = torch.index_select(self.Whr, dim=0, index=tail_part[:, 1]).unsqueeze(1)
                self.tr = torch.index_select(self.Wtr, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            
            if self.embedding_type == 'concat':
                text_head = torch.index_select(self.text_entity_embedding, dim=0, index=head_part.view(-1)).view(batch_size, negative_sample_size, -1)
                text_relation = torch.index_select(self.text_relation_embedding, dim=0, index=tail_part[:, 1]).unsqueeze(1)
                text_tail = torch.index_select(self.text_entity_embedding, dim=0, index=tail_part[:, 2]).unsqueeze(1)
            else:
                text_head = text_relation = text_tail = None
        
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(self.entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding,dim=0,index=head_part[:, 1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            
            if self.model_name == 'QuatRE':
                self.hr = torch.index_select(self.Whr, dim=0, index=head_part[:, 1]).unsqueeze(1)
                self.tr = torch.index_select(self.Wtr, dim=0, index=head_part[:, 1]).unsqueeze(1)
            
            if self.embedding_type == 'concat':
                text_head = torch.index_select(self.text_entity_embedding, dim=0, index=head_part[:, 0]).unsqueeze(1)
                text_relation = torch.index_select(self.text_relation_embedding,dim=0,index=head_part[:, 1]).unsqueeze(1)
                text_tail = torch.index_select(self.text_entity_embedding, dim=0, index=tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            else:
                text_head = text_relation = text_tail = None
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'HAKE': self.HAKE,
            'TripleRE': self.TripleRE,
            'QuatRE': self.QuatRE,
            'Rotate4D': self.Rotate4D
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode, text_head, text_relation, text_tail)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        if self.embedding_type == 'concat':
            head = torch.cat([head, text_head], dim = 2)
            relation = torch.cat([relation, text_relation], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        if self.embedding_type == 'concat':
            head = torch.cat([head, text_head], dim = 2)
            relation = torch.cat([relation, text_relation], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        if self.embedding_type == 'concat':
            re_text_head, im_text_head = torch.chunk(text_head, 2, dim=2)
            re_text_relation, im_text_relation = torch.chunk(text_relation, 2, dim=2)
            re_text_tail, im_text_tail = torch.chunk(text_tail, 2, dim=2)
            
            re_head = torch.cat([re_head, re_text_head], dim = 2)
            im_head = torch.cat([im_head, im_text_head], dim = 2)
            re_relation = torch.cat([re_relation, re_text_relation], dim = 2)
            im_relation = torch.cat([im_relation, im_text_relation], dim = 2)
            re_tail = torch.cat([re_tail, re_text_tail], dim = 2)
            im_tail = torch.cat([im_tail, im_text_tail], dim = 2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        if self.embedding_type == 'concat':
            re_text_head, im_text_head = torch.chunk(text_head, 2, dim=2)
            re_text_tail, im_text_tail = torch.chunk(text_tail, 2, dim=2)
            
            re_head = torch.cat([re_head, re_text_head], dim = 2)
            im_head = torch.cat([im_head, im_text_head], dim = 2)
            re_tail = torch.cat([re_tail, re_text_tail], dim = 2)
            im_tail = torch.cat([im_tail, im_text_tail], dim = 2)
            relation = torch.cat([relation, text_relation], dim = 2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch' == 'concat':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
    
    def HAKE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        pi = 3.14159265358979323846
        
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)
        
        if self.embedding_type == 'concat':
            phase_text_head, mod_text_head = torch.chunk(text_head, 2, dim=2)
            phase_text_relation, mod_text_relation, bias_text_relation = torch.chunk(text_relation, 3, dim=2)
            phase_text_tail, mod_text_tail = torch.chunk(text_tail, 2, dim=2)
            
            phase_head = torch.cat([phase_head, phase_text_head], dim = 2)
            mod_head = torch.cat([mod_head, mod_text_head], dim = 2)
            phase_relation = torch.cat([phase_relation, phase_text_relation], dim = 2)
            mod_relation = torch.cat([mod_relation, mod_text_relation], dim = 2)
            bias_relation = torch.cat([bias_relation, bias_text_relation], dim = 2)
            phase_tail = torch.cat([phase_tail, phase_text_tail], dim = 2)
            mod_tail = torch.cat([mod_tail, mod_text_tail], dim = 2)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        phase_tail = phase_tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
    
    def TripleRE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim = 2)
        
        if self.embedding_type == 'concat':
            re_text_head, re_text_mid, re_text_tail = torch.chunk(text_relation, 3, dim = 2)
            
            head = torch.cat([head, text_head], dim = 2)
            re_head = torch.cat([re_head, re_text_head], dim = 2)
            re_mid = torch.cat([re_mid, re_text_mid], dim = 2)
            re_tail = torch.cat([re_tail, re_text_tail], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        
        score = head * re_head - tail * re_tail + re_mid
        score = self.gamma.item() - torch.norm(score, p = 1, dim = 2)
        
        return score
    
    def Rotate4D(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        pi = 3.14159265358979323846
        
        u_h, x_h, y_h, z_h = torch.chunk(head, 4, dim = 2)
        alpha_1, alpha_2, alpha_3, bias = torch.chunk(relation, 4, dim = 2)
        u_t, x_t, y_t, z_t = torch.chunk(tail, 4, dim = 2)
        
        if self.embedding_type == 'concat':
            text_u_h, text_x_h, text_y_h, text_z_h = torch.chunk(text_head, 4, dim = 2)
            text_alpha_1, text_alpha_2, text_alpha_3, text_bias = torch.chunk(text_relation, 4, dim = 2)
            text_u_t, text_x_t, text_y_t, text_z_t = torch.chunk(text_tail, 4, dim = 2)
            
            u_h, x_h = torch.cat([u_h, text_u_h], dim = 2), torch.cat([x_h, text_x_h], dim = 2)
            y_h, z_h = torch.cat([y_h, text_y_h], dim = 2), torch.cat([z_h, text_z_h], dim = 2)
            alpha_1, alpha_2 = torch.cat([alpha_1, text_alpha_1], dim = 2), torch.cat([alpha_2, text_alpha_2], dim = 2)
            alpha_3, bias = torch.cat([alpha_3, text_alpha_3], dim = 2), torch.cat([bias, text_bias], dim = 2)
            u_t, x_t = torch.cat([u_t, text_u_t], dim = 2), torch.cat([x_t, text_x_t], dim = 2)
            y_t, z_t = torch.cat([y_t, text_y_t], dim = 2), torch.cat([z_t, text_z_t], dim = 2)
        
        bias = torch.abs(bias)
        
        # make phases of relations uniformly distributed in [-pi, pi]
        alpha_1 = alpha_1 / (self.embedding_range.item() / pi)
        alpha_2 = alpha_2 / (self.embedding_range.item() / pi)
        alpha_3 = alpha_3 / (self.embedding_range.item() / pi)
        
        # obtain representation of the rotation axis
        a_r = torch.cos(alpha_1)
        b_r = torch.sin(alpha_1) * torch.cos(alpha_2)
        c_r = torch.sin(alpha_1) * torch.sin(alpha_2) * torch.cos(alpha_3)
        d_r = torch.sin(alpha_1) * torch.sin(alpha_2) * torch.sin(alpha_3)
        
        if mode == 'head-batch':
            score_u = (a_r*u_t - b_r*x_t - c_r*y_t - d_r*z_t)*bias - u_h
            score_x = (a_r*x_t + b_r*u_t + c_r*z_t - d_r*y_t)*bias - x_h
            score_y = (a_r*y_t - b_r*z_t + c_r*u_t + d_r*x_t)*bias - y_h
            score_z = (a_r*z_t + b_r*y_t - c_r*x_t + d_r*u_t)*bias - z_h
        else:
            score_u = (a_r*u_h - b_r*x_h - c_r*y_h - d_r*z_h)*bias - u_t
            score_x = (a_r*x_h + b_r*u_h + c_r*z_h - d_r*y_h)*bias - x_t
            score_y = (a_r*y_h - b_r*z_h + c_r*u_h + d_r*x_h)*bias - y_t
            score_z = (a_r*z_h + b_r*y_h - c_r*x_h + d_r*u_h)*bias - z_t
        
        score = torch.stack([score_u, score_x, score_y, score_z], dim = 0)
        score = score.norm(dim = 0, p = 2)
        score = self.gamma.item() - score.sum(dim = 2)
        
        return score
    
    def QuatRE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        
        if self.embedding_type == 'concat':
            relation = torch.cat([relation, text_relation], dim = 2)
            h_r = self.concat_vec_vec_wise_multiplication(head, text_head, self.hr)
            t_r = self.concat_vec_vec_wise_multiplication(tail, text_tail, self.tr)
            hrr = self.vec_vec_wise_multiplication(h_r, relation)
        else:
            h_r = self.vec_vec_wise_multiplication(head, self.hr)
            t_r = self.vec_vec_wise_multiplication(tail, self.tr)
            hrr = self.vec_vec_wise_multiplication(h_r, relation)
        score = hrr * t_r
        
        return score.sum(dim = 2)
    
    def normalization(self, quaternion, split_dim=2):  # vectorized quaternion bs x 4dim
        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
        quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
        quaternion = quaternion.reshape(-1, 1, 4 * size)
        return quaternion
    
    def make_wise_quaternion(self, quaternion):  # for vector * vector quaternion element-wise multiplication
        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(2) // 4
        r, i, j, k = torch.split(quaternion, size, dim=2)
        r2 = torch.cat([r, -i, -j, -k], dim=2)  # 0, 1, 2, 3 --> bs x 4dim
        i2 = torch.cat([i, r, -k, j], dim=2)  # 1, 0, 3, 2
        j2 = torch.cat([j, k, r, -i], dim=2)  # 2, 3, 0, 1
        k2 = torch.cat([k, -j, i, r], dim=2)  # 3, 2, 1, 0
        return r2, i2, j2, k2
    
    def get_quaternion_wise_mul(self, quaternion):
        bs = quaternion.size(0)
        num_neg = quaternion.size(1)
        size = quaternion.size(2) // 4
        quaternion = quaternion.view(bs, num_neg, 4, size)
        quaternion = torch.sum(quaternion, 2)
        return quaternion

    def vec_vec_wise_multiplication(self, q, p):  # vector * vector
        normalized_p = self.normalization(p)  # bs x 4dim
        q_r, q_i, q_j, q_k = self.make_wise_quaternion(q)  # bs x 4dim

        qp_r = self.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
        qp_i = self.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
        qp_j = self.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
        qp_k = self.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=2)
    
    def concat_vec_vec_wise_multiplication(self, q, text_q, p):  # vector * vector
        normalized_p = self.normalization(p)  # bs x 4dim
        q_r, q_i, q_j, q_k = self.make_wise_quaternion(q)  # bs x 4dim
        text_q_r, text_q_i, text_q_j, text_q_k = self.make_wise_quaternion(text_q)  # bs x 4dim

        q_r, q_i = torch.cat([q_r, text_q_r], dim = 2), torch.cat([q_i, text_q_i], dim = 2)
        q_j, q_k = torch.cat([q_j, text_q_j], dim = 2), torch.cat([q_k, text_q_k], dim = 2)

        qp_r = self.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
        qp_i = self.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
        qp_j = self.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
        qp_k = self.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=2)

