# https://github.com/snap-stanford/ogb/tree/master
# RoataE: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
# ConvE: https://github.com/TimDettmers/ConvE
# ConvKB: https://github.com/daiquocnguyen/ConvKB
# CompGCN: https://github.com/malllabiisc/CompGCN

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from torch_geometric.nn import RGCNConv

from module.compgcn_layer import CompGCNConv, CompGCNConvBasis


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, num_relation_embedding=1):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.phase_weight = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[1.0]]))
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*num_relation_embedding
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'HAKE':
            nn.init.ones_(
                tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
            )
            nn.init.zeros_(
                tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
            )
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'HAKE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or num_relation_embedding != 1):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or num_relation_embedding != 2):
            raise ValueError('ComplEx should use --double_entity_embedding and --num_relation_embedding 2')
        
        if model_name == 'HAKE' and (not double_entity_embedding or num_relation_embedding != 3):
            raise ValueError('HAKE should use --double_entity_embedding and --num_relation_embedding 3')

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

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'HAKE': self.HAKE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score


    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

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


    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

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
    
    def HAKE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(relation, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

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

    def forward(self, s, r):
        embed_s = self.embed_e(s)
        embed_r = self.embed_r(r)

        embed_s = embed_s.view(-1, self.embedding_size_w, self.embedding_size_h)
        embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
        conv_input = torch.cat([embed_s, embed_r], dim=1).unsqueeze(1)
        out = self.conv_e(conv_input)

        scores = out.mm(self.embed_e.weight.t())

        return scores
    
    def valid(self, s, r, o):
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


# --------------------------------- CompGCN --------------------------------- #
class CompGCNBase(nn.Module):
    def __init__(self, hidden_dim, nentity, nrelation, edge_index, edge_type, num_bases=5, score_func='dismult', bias=False, dropout=0, opn='mult'):
        super(CompGCNBase, self).__init__()
        
        self.dropout = dropout
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.score_func = score_func
        self.init_embed = nn.Parameter(torch.Tensor(nentity, hidden_dim))
        
        if num_bases > 0:
            self.init_rel = nn.Parameter(torch.Tensor(num_bases, hidden_dim))
        else:
            if self.score_func == 'transe': 
                self.init_rel = nn.Parameter(torch.Tensor(nrelation, hidden_dim))
            else:
                self.init_rel = nn.Parameter(torch.Tensor(nrelation*2, hidden_dim)) 
        
        if num_bases > 0:
            self.conv1 = CompGCNConvBasis(hidden_dim, nrelation, num_bases, opn=opn)
            self.conv2 = CompGCNConv(hidden_dim, nrelation, opn=opn)
        else:
            self.conv1 = CompGCNConv(hidden_dim, nrelation)
            self.conv2 = CompGCNConv(hidden_dim, nrelation)
        
        if bias: self.register_parameter('bias', nn.Parameter(torch.zeros(nentity)))

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.init_rel)
        nn.init.xavier_normal_(self.init_embed)
    
    def forward(self, sub, rel, obj):
        r = self.init_rel if self.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
        
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        obj_emb = torch.index_select(x, 0, obj)
        
        return sub_emb, rel_emb, obj_emb
