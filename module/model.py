# https://github.com/snap-stanford/ogb/tree/master
# RoataE: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
# ConvE: https://github.com/TimDettmers/ConvE
# ConvKB: https://github.com/daiquocnguyen/ConvKB
# CompGCN: https://github.com/malllabiisc/CompGCN

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
        self.use_description = args.use_description
        
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
            
        self.entity_embedding = nn.Parameter(torch.zeros(args.nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
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
                tensor=self.relation_embedding[:, args.hidden_dim:2 * args.hidden_dim]
            )
            nn.init.zeros_(
                tensor=self.relation_embedding[:, 2 * args.hidden_dim:3 * args.hidden_dim]
            )
            self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
            self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]))
            
        if args.model == 'Rotate4D':
            nn.init.ones_(tensor=self.relation_embedding[:, 3*args.hidden_dim:4*args.hidden_dim])
        
        if args.model == 'QuatRE':
            if args.use_description:
                self.Whr = nn.Parameter(torch.zeros(args.nrelation, 4 * (args.hidden_dim * 2)))
                self.Wtr = nn.Parameter(torch.zeros(args.nrelation, 4 * (args.hidden_dim * 2)))
                nn.init.xavier_uniform_(self.Whr)
                nn.init.xavier_uniform_(self.Wtr)
            else:
                self.Whr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
                self.Wtr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
                nn.init.xavier_uniform_(self.Whr)
                nn.init.xavier_uniform_(self.Wtr)
        
        if args.use_description:
            self.load_embedding()
            
            self.entity_mlp = nn.Linear(self.biot5_entity_embedding.size(1), self.entity_dim)
            self.relation_mlp = nn.Linear(self.biot5_relation_embedding.size(1), self.relation_dim)
        
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
        if self.dataset == 'cd':
            self.biot5_entity_embedding = torch.load('dataset/cd/processed/biot5+_entity_embedding')
            self.biot5_relation_embedding = torch.load('dataset/cd/processed/biot5+_relation_embedding')
        elif self.dataset == 'cgd':
            self.biot5_entity_embedding = torch.load('dataset/cgd/processed/biot5+_entity_embedding')
            self.biot5_relation_embedding = torch.load('dataset/cgd/processed/biot5+_relation_embedding')
        elif self.dataset == 'cgpd':
            self.biot5_entity_embedding = torch.load('dataset/cgpd/processed/biot5+_entity_embedding')
            self.biot5_relation_embedding = torch.load('dataset/cgpd/processed/biot5+_relation_embedding')
        elif self.dataset == 'ctd':
            self.biot5_entity_embedding = torch.load('dataset/ctd/processed/biot5+_entity_embedding')
            self.biot5_relation_embedding = torch.load('dataset/ctd/processed/biot5+_relation_embedding')
        
        self.biot5_entity_embedding  = self.biot5_entity_embedding.cuda()
        self.biot5_relation_embedding  = self.biot5_relation_embedding.cuda()
    
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        if self.use_description:
            self.text_entity_embedding = self.entity_mlp(self.biot5_entity_embedding)
            self.text_relation_embedding = self.relation_mlp(self.biot5_relation_embedding)
            
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(self.entity_embedding, dim=0, index=sample[:,0]).unsqueeze(1)
            relation = torch.index_select(self.relation_embedding, dim=0, index=sample[:,1]).unsqueeze(1)
            tail = torch.index_select(self.entity_embedding, dim=0, index=sample[:,2]).unsqueeze(1)
            
            if self.model_name == 'QuatRE':
                self.hr = torch.index_select(self.Whr, dim=0, index=sample[:,1]).unsqueeze(1)
                self.tr = torch.index_select(self.Wtr, dim=0, index=sample[:,1]).unsqueeze(1)
            
            if self.use_description:
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
            
            if self.use_description:
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
            
            if self.use_description:
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
        if self.use_description:
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
        if self.use_description:
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
        
        if self.use_description:
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
        
        if self.use_description:
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

        if mode == 'head-batch':
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
        
        if self.use_description:
            phase_text_head, mod_text_head = torch.chunk(text_head, 2, dim=2)
            phase_text_relation, mod_text_relation, bias_text_relation = torch.chunk(text_relation, 3, dim=2)
            phase_text_tail, mod_text_tail = torch.chunk(text_tail, 2, dim=2)
            
            phase_head = torch.cat([phase_head, phase_text_head], dim = 2)
            mod_head = torch.cat([mod_head, mod_text_head], dim = 2)
            phase_relation = torch.cat([phase_relation, phase_text_relation], dim = 2)
            mod_relation = torch.cat([mod_relation, mod_text_relation], dim = 2)
            bias_relation = torch.cat([mod_relation, bias_text_relation], dim = 2)
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
        
        if self.use_description:
            re_text_head, re_text_mid, re_text_tail = torch.chunk(text_relation, 3, dim = 2)
            
            head = torch.cat([head, text_head], dim = 2)
            re_head = torch.cat([re_head, re_text_head], dim = 2)
            re_mid = torch.cat([re_mid, re_text_mid], dim = 2)
            re_tail = torch.cat([re_mid, re_text_tail], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        
        score = head * re_head - tail * re_tail + re_mid
        score = self.gamma.item() - torch.norm(score, p = 1, dim = 2)
        
        return score
    
    def QuatRE(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        if self.use_description:
            head = torch.cat([head, text_head], dim = 2)
            relation = torch.cat([relation, text_relation], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        h_r = self.vec_vec_wise_multiplication(head, self.hr)
        t_r = self.vec_vec_wise_multiplication(tail, self.tr)
        hrr = self.vec_vec_wise_multiplication(h_r, relation)
        score = hrr * t_r
        
        return score.sum(dim = 2)
    
    def Rotate4D(self, head, relation, tail, mode, text_head, text_relation, text_tail):
        pi = 3.14159265358979323846
        
        if self.use_description:
            head = torch.cat([head, text_head], dim = 2)
            relation = torch.cat([relation, text_relation], dim = 2)
            tail = torch.cat([tail, text_tail], dim = 2)
        
        u_h, x_h, y_h, z_h = torch.chunk(head, 4, dim = 2)
        alpha_1, alpha_2, alpha_3, bias = torch.chunk(relation, 4, dim = 2)
        u_t, x_t, y_t, z_t = torch.chunk(tail, 4, dim = 2)
        
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


# class KGEModel(nn.Module):
#     def __init__(self, args):
#         super(KGEModel, self).__init__()
        
#         self.dataset = args.dataset
#         self.model_name = args.model
#         self.use_description = args.use_description
        
#         self.nentity = args.nentity
#         self.nrelation = args.nrelation
#         self.hidden_dim = args.hidden_dim
#         self.epsilon = 2.0
        
#         self.gamma = nn.Parameter(
#             torch.Tensor([args.gamma]), 
#             requires_grad=False
#         )
        
#         self.entity_dim = args.hidden_dim * args.num_entity_embedding
#         self.relation_dim = args.hidden_dim * args.num_relation_embedding
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / args.hidden_dim]), 
#             requires_grad=False
#         )
        
#         if args.use_description:
#             self.load_embedding()
            
#             self.entity_mlp = nn.Linear(self.biot5_entity_embedding.size(1), self.entity_dim)
#             self.relation_mlp = nn.Linear(self.biot5_relation_embedding.size(1), self.relation_dim)
            
#             if args.model == 'HAKE':
#                 self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
#                 self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]))
            
#         else:
#             self.entity_embedding = nn.Parameter(torch.zeros(args.nentity, self.entity_dim))
#             nn.init.uniform_(
#                 tensor=self.entity_embedding, 
#                 a=-self.embedding_range.item(), 
#                 b=self.embedding_range.item()
#             )
            
#             self.relation_embedding = nn.Parameter(torch.zeros(args.nrelation, self.relation_dim))
#             nn.init.uniform_(
#                 tensor=self.relation_embedding, 
#                 a=-self.embedding_range.item(), 
#                 b=self.embedding_range.item()
#             )
        
#             if args.model == 'HAKE':
#                 nn.init.ones_(
#                     tensor=self.relation_embedding[:, args.hidden_dim:2 * args.hidden_dim]
#                 )
#                 nn.init.zeros_(
#                     tensor=self.relation_embedding[:, 2 * args.hidden_dim:3 * args.hidden_dim]
#                 )
#                 self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
#                 self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]))
                
#             if args.model == 'Rotate4D':
#                 nn.init.ones_(tensor=self.relation_embedding[:, 3*args.hidden_dim:4*args.hidden_dim])
        
#         if args.model == 'QuatRE':
#             self.Whr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
#             self.Wtr = nn.Parameter(torch.zeros(args.nrelation, 4 * args.hidden_dim))
#             nn.init.xavier_uniform_(self.Whr)
#             nn.init.xavier_uniform_(self.Wtr)
        
#         #Do not forget to modify this line when you add a new model in the "forward" function
#         if args.model not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'HAKE', 'TripleRE', 'QuatRE', 'Rotate4D']:
#             raise ValueError('model %s not supported' % args.model)
            
#         if args.model == 'RotatE' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 1):
#             raise ValueError('RotatE should use --num_entity_embedding 2')

#         if args.model == 'ComplEx' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 2):
#             raise ValueError('ComplEx should use --num_entity_embedding 2 and --num_relation_embedding 2')
        
#         if args.model == 'HAKE' and (args.num_entity_embedding != 2 or args.num_relation_embedding != 3):
#             raise ValueError('HAKE should use --num_entity_embedding 2 and --num_relation_embedding 3')
        
#         if args.model == 'TripleRE' and (args.num_relation_embedding != 3):
#             raise ValueError('TripleRE should use --num_relation_embedding 3')
        
#         if (args.model == 'QuatRE' or args.model == 'Rotate4D') and (args.num_entity_embedding != 4 or args.num_relation_embedding != 4):
#             raise ValueError('QuatRE or Rotate4D should use --num_entity_embedding 4 and --num_relation_embedding 4')

#     def load_embedding(self):
#         if self.dataset == 'cd':
#             self.biot5_entity_embedding = torch.load('dataset/cd/processed/entity_embedding')
#             self.biot5_relation_embedding = torch.load('dataset/cd/processed/relation_embedding')
#         # elif self.dataset == 'cgd':
#         # elif self.dataset == 'cgpd':
#         # elif self.dataset == 'ctd':
#         self.biot5_entity_embedding  = self.biot5_entity_embedding.cuda()
#         self.biot5_relation_embedding  = self.biot5_relation_embedding.cuda()
    
#     def forward(self, sample, mode='single'):
#         '''
#         Forward function that calculate the score of a batch of triples.
#         In the 'single' mode, sample is a batch of triple.
#         In the 'head-batch' or 'tail-batch' mode, sample consists two part.
#         The first part is usually the positive sample.
#         And the second part is the entities in the negative samples.
#         Because negative samples and positive samples usually share two elements 
#         in their triple ((head, relation) or (relation, tail)).
#         '''
#         if self.use_description:
#             self.entity_embedding = self.entity_mlp(self.biot5_entity_embedding)
#             self.relation_embedding = self.relation_mlp(self.biot5_relation_embedding)
            
#         if mode == 'single':
#             batch_size, negative_sample_size = sample.size(0), 1
            
#             head = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=sample[:,0]
#             ).unsqueeze(1)
            
#             relation = torch.index_select(
#                 self.relation_embedding, 
#                 dim=0, 
#                 index=sample[:,1]
#             ).unsqueeze(1)
            
#             tail = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=sample[:,2]
#             ).unsqueeze(1)
            
#             if self.model_name == 'QuatRE':
#                 self.hr = torch.index_select(self.Whr, dim=0, index=sample[:,1]).unsqueeze(1)
#                 self.tr = torch.index_select(self.Wtr, dim=0, index=sample[:,1]).unsqueeze(1)
            
#         elif mode == 'head-batch':
#             tail_part, head_part = sample
#             batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
#             head = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=head_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
            
#             relation = torch.index_select(
#                 self.relation_embedding, 
#                 dim=0, 
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)
            
#             tail = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=tail_part[:, 2]
#             ).unsqueeze(1)
            
#             if self.model_name == 'QuatRE':
#                 self.hr = torch.index_select(self.Whr, dim=0, index=tail_part[:, 1]).unsqueeze(1)
#                 self.tr = torch.index_select(self.Wtr, dim=0, index=tail_part[:, 1]).unsqueeze(1)
            
#         elif mode == 'tail-batch':
#             head_part, tail_part = sample
#             batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
#             head = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=head_part[:, 0]
#             ).unsqueeze(1)
            
#             relation = torch.index_select(
#                 self.relation_embedding,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)
            
#             tail = torch.index_select(
#                 self.entity_embedding, 
#                 dim=0, 
#                 index=tail_part.view(-1)
#             ).view(batch_size, negative_sample_size, -1)
            
#             if self.model_name == 'QuatRE':
#                 self.hr = torch.index_select(self.Whr, dim=0, index=head_part[:, 1]).unsqueeze(1)
#                 self.tr = torch.index_select(self.Wtr, dim=0, index=head_part[:, 1]).unsqueeze(1)
            
#         else:
#             raise ValueError('mode %s not supported' % mode)
        
        
#         model_func = {
#             'TransE': self.TransE,
#             'DistMult': self.DistMult,
#             'ComplEx': self.ComplEx,
#             'RotatE': self.RotatE,
#             'HAKE': self.HAKE,
#             'TripleRE': self.TripleRE,
#             'QuatRE': self.QuatRE,
#             'Rotate4D': self.Rotate4D
#         }
        
#         if self.model_name in model_func:
#             score = model_func[self.model_name](head, relation, tail, mode)
#         else:
#             raise ValueError('model %s not supported' % self.model_name)
        
#         return score
    
#     def TransE(self, head, relation, tail, mode):
#         if mode == 'head-batch':
#             score = head + (relation - tail)
#         else:
#             score = (head + relation) - tail

#         score = self.gamma.item() - torch.norm(score, p=1, dim=2)
#         return score

#     def DistMult(self, head, relation, tail, mode):
#         if mode == 'head-batch':
#             score = head * (relation * tail)
#         else:
#             score = (head * relation) * tail

#         score = score.sum(dim = 2)
#         return score

#     def ComplEx(self, head, relation, tail, mode):
#         re_head, im_head = torch.chunk(head, 2, dim=2)
#         re_relation, im_relation = torch.chunk(relation, 2, dim=2)
#         re_tail, im_tail = torch.chunk(tail, 2, dim=2)

#         if mode == 'head-batch':
#             re_score = re_relation * re_tail + im_relation * im_tail
#             im_score = re_relation * im_tail - im_relation * re_tail
#             score = re_head * re_score + im_head * im_score
#         else:
#             re_score = re_head * re_relation - im_head * im_relation
#             im_score = re_head * im_relation + im_head * re_relation
#             score = re_score * re_tail + im_score * im_tail

#         score = score.sum(dim = 2)
#         return score

#     def RotatE(self, head, relation, tail, mode):
#         pi = 3.14159265358979323846
        
#         re_head, im_head = torch.chunk(head, 2, dim=2)
#         re_tail, im_tail = torch.chunk(tail, 2, dim=2)

#         #Make phases of relations uniformly distributed in [-pi, pi]

#         phase_relation = relation/(self.embedding_range.item()/pi)

#         re_relation = torch.cos(phase_relation)
#         im_relation = torch.sin(phase_relation)

#         if mode == 'head-batch':
#             re_score = re_relation * re_tail + im_relation * im_tail
#             im_score = re_relation * im_tail - im_relation * re_tail
#             re_score = re_score - re_head
#             im_score = im_score - im_head
#         else:
#             re_score = re_head * re_relation - im_head * im_relation
#             im_score = re_head * im_relation + im_head * re_relation
#             re_score = re_score - re_tail
#             im_score = im_score - im_tail

#         score = torch.stack([re_score, im_score], dim = 0)
#         score = score.norm(dim = 0)

#         score = self.gamma.item() - score.sum(dim = 2)
#         return score
    
#     def HAKE(self, head, relation, tail, mode):
#         pi = 3.14159265358979323846
        
#         phase_head, mod_head = torch.chunk(head, 2, dim=2)
#         phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=2)
#         phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

#         phase_head = phase_head / (self.embedding_range.item() / pi)
#         phase_relation = phase_relation / (self.embedding_range.item() / pi)
#         phase_tail = phase_tail / (self.embedding_range.item() / pi)

#         if mode == 'head-batch':
#             phase_score = phase_head + (phase_relation - phase_tail)
#         else:
#             phase_score = (phase_head + phase_relation) - phase_tail

#         mod_relation = torch.abs(mod_relation)
#         bias_relation = torch.clamp(bias_relation, max=1)
#         indicator = (bias_relation < -mod_relation)
#         bias_relation[indicator] = -mod_relation[indicator]

#         r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

#         phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
#         r_score = torch.norm(r_score, dim=2) * self.modulus_weight

#         return self.gamma.item() - (phase_score + r_score)
    
#     def TripleRE(self, head, relation, tail, mode):
#         re_head, re_mid, re_tail = torch.chunk(relation, 3, dim = 2)
        
#         head = F.normalize(head, 2, -1)
#         tail = F.normalize(tail, 2, -1)
        
#         score = head * re_head - tail * re_tail + re_mid
#         score = self.gamma.item() - torch.norm(score, p = 1, dim = 2)
        
#         return score
    
#     def QuatRE(self, head, relation, tail, mode):
#         h_r = self.vec_vec_wise_multiplication(head, self.hr)
#         t_r = self.vec_vec_wise_multiplication(tail, self.tr)
#         hrr = self.vec_vec_wise_multiplication(h_r, relation)
#         score = hrr * t_r
        
#         return score.sum(dim = 2)
    
#     def Rotate4D(self, head, relation, tail, mode):
#         pi = 3.14159265358979323846
        
#         u_h, x_h, y_h, z_h = torch.chunk(head, 4, dim = 2)
#         alpha_1, alpha_2, alpha_3, bias = torch.chunk(relation, 4, dim = 2)
#         u_t, x_t, y_t, z_t = torch.chunk(tail, 4, dim = 2)
        
#         bias = torch.abs(bias)
        
#         # make phases of relations uniformly distributed in [-pi, pi]
#         alpha_1 = alpha_1 / (self.embedding_range.item() / pi)
#         alpha_2 = alpha_2 / (self.embedding_range.item() / pi)
#         alpha_3 = alpha_3 / (self.embedding_range.item() / pi)
        
#         # obtain representation of the rotation axis
#         a_r = torch.cos(alpha_1)
#         b_r = torch.sin(alpha_1) * torch.cos(alpha_2)
#         c_r = torch.sin(alpha_1) * torch.sin(alpha_2) * torch.cos(alpha_3)
#         d_r = torch.sin(alpha_1) * torch.sin(alpha_2) * torch.sin(alpha_3)
        
#         if mode == 'head-batch':
#             score_u = (a_r*u_t - b_r*x_t - c_r*y_t - d_r*z_t)*bias - u_h
#             score_x = (a_r*x_t + b_r*u_t + c_r*z_t - d_r*y_t)*bias - x_h
#             score_y = (a_r*y_t - b_r*z_t + c_r*u_t + d_r*x_t)*bias - y_h
#             score_z = (a_r*z_t + b_r*y_t - c_r*x_t + d_r*u_t)*bias - z_h
#         else:
#             score_u = (a_r*u_h - b_r*x_h - c_r*y_h - d_r*z_h)*bias - u_t
#             score_x = (a_r*x_h + b_r*u_h + c_r*z_h - d_r*y_h)*bias - x_t
#             score_y = (a_r*y_h - b_r*z_h + c_r*u_h + d_r*x_h)*bias - y_t
#             score_z = (a_r*z_h + b_r*y_h - c_r*x_h + d_r*u_h)*bias - z_t
        
#         score = torch.stack([score_u, score_x, score_y, score_z], dim = 0)
#         score = score.norm(dim = 0, p = 2)
#         score = self.gamma.item() - score.sum(dim = 2)
        
#         return score
    
#     def normalization(self, quaternion, split_dim=2):  # vectorized quaternion bs x 4dim
#         size = quaternion.size(split_dim) // 4
#         quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
#         quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
#         quaternion = quaternion.reshape(-1, 1, 4 * size)
#         return quaternion
    
#     def make_wise_quaternion(self, quaternion):  # for vector * vector quaternion element-wise multiplication
#         if len(quaternion.size()) == 1:
#             quaternion = quaternion.unsqueeze(0)
#         size = quaternion.size(2) // 4
#         r, i, j, k = torch.split(quaternion, size, dim=2)
#         r2 = torch.cat([r, -i, -j, -k], dim=2)  # 0, 1, 2, 3 --> bs x 4dim
#         i2 = torch.cat([i, r, -k, j], dim=2)  # 1, 0, 3, 2
#         j2 = torch.cat([j, k, r, -i], dim=2)  # 2, 3, 0, 1
#         k2 = torch.cat([k, -j, i, r], dim=2)  # 3, 2, 1, 0
#         return r2, i2, j2, k2
    
#     def get_quaternion_wise_mul(self, quaternion):
#         bs = quaternion.size(0)
#         num_neg = quaternion.size(1)
#         size = quaternion.size(2) // 4
#         quaternion = quaternion.view(bs, num_neg, 4, size)
#         quaternion = torch.sum(quaternion, 2)
#         return quaternion

#     def vec_vec_wise_multiplication(self, q, p):  # vector * vector
#         normalized_p = self.normalization(p)  # bs x 4dim
#         q_r, q_i, q_j, q_k = self.make_wise_quaternion(q)  # bs x 4dim

#         qp_r = self.get_quaternion_wise_mul(q_r * normalized_p)  # qrpr−qipi−qjpj−qkpk
#         qp_i = self.get_quaternion_wise_mul(q_i * normalized_p)  # qipr+qrpi−qkpj+qjpk
#         qp_j = self.get_quaternion_wise_mul(q_j * normalized_p)  # qjpr+qkpi+qrpj−qipk
#         qp_k = self.get_quaternion_wise_mul(q_k * normalized_p)  # qkpr−qjpi+qipj+qrpk

#         return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=2)


# --------------------------------- RGCN --------------------------------- #
class GNNEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, gnn_model, num_layers, dropout = 0.3):
        super(GNNEncoder, self).__init__()
        
        self.node_emb = nn.Parameter(torch.Tensor(num_nodes, hidden_dim))
        self.gnn_model = gnn_model
        self.dropout = dropout

        if gnn_model == 'rgcn':
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_blocks=5))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb)
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, edge_index, edge_type):
        x = self.node_emb
        
        if self.gnn_model == 'rgcn':
            for conv in self.convs:
                x = conv(x, edge_index, edge_type)
                x = nn.functional.relu(x)
                x = nn.functional.dropout(x, p = self.dropout, training = self.training)
            x = self.convs[-1](x, edge_index, edge_type)
        
        return x


class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, hidden_dim):
        super(DistMultDecoder, self).__init__()
        
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb)
    
    def forward(self, h, r, t, mode):
        r = self.rel_emb[r]
        if mode == 'head-batch':
            score = h * (r * t)
        else:
            score = (h * r) * t

        return torch.sum(score, dim = 1)


# --------------------------------- ConvE --------------------------------- #
class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        x = x.view(n, -1)
        return x


class ConvE(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, embedding_size_w=10,
                 conv_channels=32, conv_kernel_size=3, embed_dropout=0.2, feature_map_dropout=0.2,
                 proj_layer_dropout=0.3):
        super().__init__()

        self.num_e = num_nodes
        self.num_r = num_relations
        self.embedding_size_h = hidden_dim//embedding_size_w
        self.embedding_size_w = embedding_size_w

        flattened_size = (embedding_size_w * 2 - conv_kernel_size + 1) * \
                         (self.embedding_size_h - conv_kernel_size + 1) * conv_channels

        self.embed_e = nn.Embedding(num_embeddings=self.num_e, embedding_dim=hidden_dim)
        self.embed_r = nn.Embedding(num_embeddings=self.num_r, embedding_dim=hidden_dim)

        self.conv_e = nn.Sequential(
            nn.Dropout(p=embed_dropout),
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.Dropout2d(p=feature_map_dropout),

            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Dropout(p=proj_layer_dropout)
        )
        self.init()
        
    def init(self):
        xavier_normal_(self.embed_e.weight.data)
        xavier_normal_(self.embed_r.weight.data)

    # def forward(self, s, r):
    #     embed_s = self.embed_e(s)
    #     embed_r = self.embed_r(r)

    #     embed_s = embed_s.view(-1, self.embedding_size_w, self.embedding_size_h)
    #     embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
    #     conv_input = torch.cat([embed_s, embed_r], dim=1).unsqueeze(1)
    #     out = self.conv_e(conv_input)

    #     scores = out.mm(self.embed_e.weight.t())

    #     return scores
    
    def forward(self, s, r, o):
        embed_s = self.embed_e(s)
        embed_o = self.embed_e(o)
        embed_r = self.embed_r(r)
        
        embed_s = embed_s.view(-1, self.embedding_size_w, self.embedding_size_h)
        embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
        conv_input = torch.cat([embed_s, embed_r], dim=1).unsqueeze(1)
        out = self.conv_e(conv_input)
        
        scores = torch.sum(out*embed_o, dim=-1)
        return scores


# --------------------------------- ConvKB --------------------------------- #
class ConvKB(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, out_channels=32, kernel_size=1, dropout=0.3):
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


# # --------------------------------- GIE --------------------------------- #

# class GIE(nn.Module):
#     def __init__(self, nentity, nrelation, hidden_dim, gamma, bias, init_size):
#         super(GIE, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.bias = bias
#         self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
#         self.entity_embedding = nn.Embedding(nentity, hidden_dim)
#         self.relation_embedding = nn.Embedding(nrelation, hidden_dim)
#         self.bh = nn.Embedding(nentity, 1)
#         self.bt = nn.Embedding(nentity, 1)

#         self.c = nn.Parameter(torch.ones((nrelation, 1), dtype=torch.float), requires_grad=True)
#         self.c1= nn.Parameter(torch.ones((nrelation, 1), dtype=torch.float), requires_grad=True)
#         self.c2 = nn.Parameter(torch.ones((nrelation, 1), dtype=torch.float), requires_grad=True)

#         self.rel_diag = nn.Embedding(nrelation, 2 * self.hidden_dim)
#         self.rel_diag1 = nn.Embedding(nrelation, self.hidden_dim)
#         self.rel_diag2 = nn.Embedding(nrelation, self.hidden_dim)
#         self.context_vec = nn.Embedding(nrelation, self.hidden_dim)
        
#         self.bh.weight.data = torch.zeros((nentity, 1), dtype=torch.float)
#         self.bt.weight.data = torch.zeros((nentity, 1), dtype=torch.float)

#         self.entity_embedding.weight.data = init_size * torch.randn((nentity, self.hidden_dim), dtype=torch.float)
#         self.relation_embedding.weight.data = init_size * torch.randn((nrelation, 2 * self.hidden_dim), dtype=torch.float)
#         self.rel_diag.weight.data = 2 * torch.rand((nrelation, 2 * self.hidden_dim), dtype=torch.float) - 1.0
#         self.context_vec.weight.data = init_size * torch.randn((nrelation, self.hidden_dim), dtype=torch.float)
#         self.scale = nn.Parameter(torch.Tensor([1. / (self.hidden_dim**0.5)]),requires_grad=False)

#     def similarity_score(self, lhs_e, rhs_e):
#         lhs_e, c = lhs_e
#         return - hyp_distance_multi_c(lhs_e, rhs_e, c, False) ** 2
    
#     def score(self, lhs, rhs):
#         lhs_e, lhs_biases = lhs
#         rhs_e, rhs_biases = rhs
#         score = self.similarity_score(lhs_e, rhs_e)
#         if self.bias == 'constant':
#             return self.gamma.item() + score
#         elif self.bias == 'learn':
#             return lhs_biases + rhs_biases + score
#         else:
#             return score

#     def get_queries(self, head_idx, relation_idx):

#         c1 = F.softplus(self.c1[relation_idx])
#         head1 = expmap0(self.entity_embedding(head_idx), c1)
#         rel1, rel2 = torch.chunk(self.relation_embedding(relation_idx), 2, dim=1)
#         rel1 = expmap0(rel1, c1)
#         rel2 = expmap0(rel2, c1)
#         lhs = project(mobius_add(head1, rel1, c1), c1)
#         res1 = givens_rotations(self.rel_diag1(relation_idx), lhs)
#         c2 = F.softplus(self.c2[relation_idx]) 
#         head2 = expmap1(self.entity_embedding(head_idx), c2)
#         rel1, rel2 = torch.chunk(self.relation_embedding(relation_idx), 2, dim=1)
#         rel11 = expmap1(rel1, c2)
#         rel21= expmap1(rel2, c2)
#         lhss = project(mobius_add(head2, rel11, c2), c2)
#         res11 = givens_rotations(self.rel_diag2(relation_idx), lhss)
#         res1=logmap1(res1,c1)  
#         res11=logmap1(res11,c2) 
#         c = F.softplus(self.c[relation_idx])
        
#         rot_mat, _ = torch.chunk(self.rel_diag(relation_idx), 2, dim=1)
#         rot_q = givens_rotations(rot_mat, self.entity_embedding(head_idx)).view((-1, 1, self.hidden_dim))
#         cands = torch.cat([res1.view(-1, 1, self.hidden_dim),res11.view(-1, 1, self.hidden_dim),rot_q], dim=1)
#         context_vec = self.context_vec(relation_idx).view((-1, 1, self.hidden_dim))
#         att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
#         att_weights = nn.Softmax(dim=-1)(att_weights)
#         att_q = torch.sum(att_weights * cands, dim=1)
#         lhs = expmap0(att_q, c)
#         rel, _ = torch.chunk(self.relation_embedding(relation_idx), 2, dim=1)
#         rel = expmap0(rel, c)
#         res = project(mobius_add(lhs, rel, c), c)
#         return (res, c), self.bh(head_idx)
    
#     def forward(self, sample, mode='single'):

#         if mode == 'single':
#             batch_size, negative_sample_size = sample.size(0), 1
            
#             head_idx = sample[:,0]
#             relation_idx = sample[:,1]
#             tail_idx = sample[:,2]
            
#         elif mode == 'head-batch':
#             tail_part, head_part = sample
#             batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
#             head_idx = head_part.view(-1)
#             relation_idx = tail_part[:, 1].repeat(negative_sample_size)
#             tail_idx = tail_part[:, 2].repeat(negative_sample_size)
            
#         elif mode == 'tail-batch':
#             head_part, tail_part = sample
#             batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
#             head_idx = head_part[:, 0].repeat(negative_sample_size)
#             relation_idx = head_part[:, 1].repeat(negative_sample_size)
#             tail_idx = tail_part.view(-1)   
        
#         lhs_e, lhs_biases = self.get_queries(head_idx, relation_idx)
#         rhs_e, rhs_biases = self.entity_embedding(tail_idx), self.bt(tail_idx)
#         score = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))
        
#         return score.view(-1, negative_sample_size)

# # --------------------------------- HousE --------------------------------- #

# class HousE(nn.Module):
#     def __init__(self, nentity, nrelation, hidden_dim, gamma, 
#                  house_dim=2, housd_num=1, thred=0.5):
#         super().__init__()
#         if house_dim % 2 == 0:
#             house_num = house_dim
#         else:
#             house_num = house_dim-1
#         self.nentity = nentity
#         self.nrelation = nrelation
#         self.hidden_dim = int(hidden_dim / house_dim)
#         self.house_dim = house_dim
#         self.housd_num = housd_num
#         self.epsilon = 2.0
#         self.thred = thred
#         self.house_num = house_num + (2*self.housd_num)

#         self.gamma = nn.Parameter(
#             torch.Tensor([gamma]), 
#             requires_grad=False
#         )
        
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
#             requires_grad=False
#         )
        
#         self.entity_dim = self.hidden_dim
#         self.relation_dim = self.hidden_dim
        
#         self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, self.house_dim))
#         nn.init.uniform_(
#             tensor=self.entity_embedding, 
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
        
#         self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim*self.house_num))
#         nn.init.uniform_(
#             tensor=self.relation_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )

#         self.k_dir_head = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
#         nn.init.uniform_(
#             tensor=self.k_dir_head,
#             a=-0.01,
#             b=+0.01
#         )

#         self.k_dir_tail = nn.Parameter(torch.zeros(nrelation, 1, self.housd_num))
#         with torch.no_grad():
#             self.k_dir_tail.data = - self.k_dir_head.data
        
#         self.k_scale_head = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
#         nn.init.uniform_(
#             tensor=self.k_scale_head,
#             a=-1,
#             b=+1
#         )

#         self.k_scale_tail = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.housd_num))
#         nn.init.uniform_(
#             tensor=self.k_scale_tail,
#             a=-1,
#             b=+1
#         )

#         self.relation_weight = nn.Parameter(torch.zeros(nrelation, self.relation_dim, self.house_dim))
#         nn.init.uniform_(
#             tensor=self.relation_weight,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )

#     def norm_embedding(self, mode):
#         entity_embedding = self.entity_embedding
#         r_list = torch.chunk(self.relation_embedding, self.house_num, 2)
#         normed_r_list = []
#         for i in range(self.house_num):
#             r_i = torch.nn.functional.normalize(r_list[i], dim=2, p=2)
#             normed_r_list.append(r_i)
#         r = torch.cat(normed_r_list, dim=2)
#         self.k_head = self.k_dir_head * torch.abs(self.k_scale_head)
#         self.k_head[self.k_head>self.thred] = self.thred
#         self.k_tail = self.k_dir_tail * torch.abs(self.k_scale_tail)
#         self.k_tail[self.k_tail>self.thred] = self.thred
#         return entity_embedding, r

#     def forward(self, sample, mode='single'):

#         entity_embedding, r = self.norm_embedding(mode)

#         if mode == 'head-batch':
#             tail_part, head_part = sample
#             batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

#             head = torch.index_select(
#                 entity_embedding,
#                 dim=0,
#                 index=head_part.view(-1)
#             ).view(batch_size, negative_sample_size, self.entity_dim, -1)

#             k_head = torch.index_select(
#                 self.k_head,
#                 dim=0,
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)

#             k_tail = torch.index_select(
#                 self.k_tail,
#                 dim=0,
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)

#             relation = torch.index_select(
#                 r,
#                 dim=0,
#                 index=tail_part[:, 1]
#             ).unsqueeze(1)

#             tail = torch.index_select(
#                 entity_embedding,
#                 dim=0,
#                 index=tail_part[:, 2]
#             ).unsqueeze(1)

#         elif mode == 'tail-batch':
#             head_part, tail_part = sample
#             batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

#             head = torch.index_select(
#                 entity_embedding,
#                 dim=0,
#                 index=head_part[:, 0]
#             ).unsqueeze(1)

#             k_head = torch.index_select(
#                 self.k_head,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)

#             k_tail = torch.index_select(
#                 self.k_tail,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)

#             relation = torch.index_select(
#                 r,
#                 dim=0,
#                 index=head_part[:, 1]
#             ).unsqueeze(1)

#             tail = torch.index_select(
#                 entity_embedding,
#                 dim=0,
#                 index=tail_part.view(-1)
#             ).view(batch_size, negative_sample_size, self.entity_dim, -1)

#         else:
#             raise ValueError('mode %s not supported' % mode)

#         r_list = torch.chunk(relation, self.house_num, 3)

#         if mode == 'head-batch':
#             for i in range(self.housd_num):
#                 k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
#                 tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

#             for i in range(self.housd_num, self.house_num-self.housd_num):
#                 tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            
#             for i in range(self.housd_num):
#                 k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
#                 head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]

#             cos_score = tail - head
#             cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)
#         else:
#             for i in range(self.housd_num):
#                 k_head_i = k_head[:, :, :, i].unsqueeze(dim=3)
#                 head = head - (0 + k_head_i) * (r_list[self.house_num-1-i] * head).sum(dim=-1, keepdim=True) * r_list[self.house_num-1-i]
            
#             for i in range(self.housd_num, self.house_num-self.housd_num):
#                 j = self.house_num - 1 - i
#                 head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            
#             for i in range(self.housd_num):
#                 k_tail_i = k_tail[:, :, :, i].unsqueeze(dim=3)
#                 tail = tail - (0 + k_tail_i) * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]

#             cos_score = head - tail
#             cos_score = torch.sum(cos_score.norm(dim=3, p=2), dim=2)

#         score = self.gamma.item() - (cos_score)
#         return score

# --------------------------------- CompGCN --------------------------------- #
# class CompGCNBase(nn.Module):
#     def __init__(self, hidden_dim, nentity, nrelation, edge_index, edge_type, num_bases=5, score_func='dismult', bias=False, dropout=0, opn='mult'):
#         super(CompGCNBase, self).__init__()
        
#         self.dropout = dropout
#         self.edge_index = edge_index
#         self.edge_type = edge_type
#         self.score_func = score_func
#         self.init_embed = nn.Parameter(torch.Tensor(nentity, hidden_dim))
        
#         if num_bases > 0:
#             self.init_rel = nn.Parameter(torch.Tensor(num_bases, hidden_dim))
#         else:
#             if self.score_func == 'transe': 
#                 self.init_rel = nn.Parameter(torch.Tensor(nrelation, hidden_dim))
#             else:
#                 self.init_rel = nn.Parameter(torch.Tensor(nrelation*2, hidden_dim)) 
        
#         if num_bases > 0:
#             self.conv1 = CompGCNConvBasis(hidden_dim, nrelation, num_bases, opn=opn)
#             self.conv2 = CompGCNConv(hidden_dim, nrelation, opn=opn)
#         else:
#             self.conv1 = CompGCNConv(hidden_dim, nrelation)
#             self.conv2 = CompGCNConv(hidden_dim, nrelation)
        
#         if bias: self.register_parameter('bias', nn.Parameter(torch.zeros(nentity)))

#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.xavier_normal_(self.init_rel)
#         nn.init.xavier_normal_(self.init_embed)
    
#     def forward(self, sub, rel, obj):
#         r = self.init_rel if self.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
#         x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
#         x = nn.functional.dropout(x, p=self.dropout, training=self.training)
#         x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
        
#         sub_emb = torch.index_select(x, 0, sub)
#         rel_emb = torch.index_select(r, 0, rel)
#         obj_emb = torch.index_select(x, 0, obj)
        
#         return sub_emb, rel_emb, obj_emb

# %%
