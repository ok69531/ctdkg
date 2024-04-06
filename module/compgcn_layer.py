import inspect, torch
from torch import nn
from torch_scatter import scatter, scatter_add


def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
	r"""Base class for creating message passing layers

	.. math::
		\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
		\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
		\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

	where :math:`\square` denotes a differentiable, permutation invariant
	function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
	and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
	MLPs.
	See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
	create_gnn.html>`__ for the accompanying tutorial.

	"""

	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()

		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

	def propagate(self, aggr, edge_index, **kwargs):
		r"""The initial call to start propagating messages.
		Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
		:obj:`"max"`), the edge indices, and all additional data which is
		needed to construct messages and to update node embeddings."""

		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index


		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x']
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args)
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
		for each edge in :math:`(i,j) \in \mathcal{E}`.
		Can take any argument which was initially passed to :meth:`propagate`.
		In addition, features can be lifted to the source node :math:`i` and
		target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
		variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

		return x_j

	def update(self, aggr_out):  # pragma: no cover
		r"""Updates node embeddings in analogy to
		:math:`\gamma_{\mathbf{\Theta}}` for each node
		:math:`i \in \mathcal{V}`.
		Takes in the output of aggregation as first argument and any argument
		which was initially passed to :meth:`propagate`."""

		return aggr_out


class CompGCNConv(MessagePassing):
    def __init__(self, hidden_dim, num_relations, dropout=0, bias=False, opn='mult', act=lambda x:x):
        super(CompGCNConv, self).__init__()
        
        # self.dropout = dropout
        self.bias = bias
        self.opn = opn
        self.hidden_dim = hidden_dim
        self.num_rels = num_relations
        self.act = act
        
        self.w_loop = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_in = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_out = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_rel = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.loop_rel = nn.Parameter(torch.Tensor(1, hidden_dim))
        
        self.drop = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        
        if self.bias: self.register_parameter('bias', nn.Parameter(torch.zeros(hidden_dim)))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_loop)
        nn.init.xavier_normal_(self.w_in)
        nn.init.xavier_normal_(self.w_out)
        nn.init.xavier_normal_(self.w_rel)
        nn.init.xavier_normal_(self.loop_rel)
    
    def forward(self, x, edge_index, edge_type, rel_embed): 
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]
        
        self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(edge_index.device)
        self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(edge_index.device)
        
        self.in_norm     = self.compute_norm(self.in_index,  num_ent)
        self.out_norm    = self.compute_norm(self.out_index, num_ent)
        
        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=self.in_norm,	mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
        out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
        
        if self.bias: out = out + self.bias
        out = self.bn(out)
        
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted
    
    def com_mult(self, a, b):
        r1, i1 = a[..., 0], a[..., 1]
        r2, i2 = b[..., 0], b[..., 1]
        return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)
    
    def conj(self, a):
        a[..., 1] = -a[..., 1]
        return a
    
    def ccorr(self, a, b):
        return torch.irfft(self.com_mult(self.conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
    
    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == 'corr': trans_embed  = self.ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub': trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult': trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError
        return trans_embed
    
    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	= torch.mm(xj_rel, weight)
        
        return out if edge_norm is None else out * edge_norm.view(-1, 1)
    
    def update(self, aggr_out):
        return aggr_out
    
    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}
        
        return norm
    
    def __repr__(self):
        return '{}({}, num_rels={})'.format(
			self.__class__.__name__, self.hidden_dim, self.num_rels)


class CompGCNConvBasis(MessagePassing):
    def __init__(self, hidden_dim, num_rels, num_bases=5, act=lambda x:x, cache=True, bias=False, dropout=0, opn='mult'):
        super(CompGCNConvBasis, self).__init__()

        self.opn = opn
        self.bias = bias
        self.hidden_dim = hidden_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.act = act
        self.cache = cache			# Should be False for graph classification tasks

        self.w_loop = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_in = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.w_out = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        self.rel_basis = nn.Parameter(torch.Tensor(self.num_bases, hidden_dim))
        self.rel_wt = nn.Parameter(torch.Tensor(self.num_rels*2, self.num_bases))
        self.w_rel = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.loop_rel = nn.Parameter(torch.Tensor(1, hidden_dim))

        self.drop = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)

        self.in_norm, self.out_norm = None, None
        self.in_index, self.out_index = None, None
        self.in_type, self.out_type = None, None
        self.loop_index, self.loop_type = None, None

        if self.bias: self.register_parameter('bias', nn.Parameter(torch.zeros(hidden_dim)))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_loop)
        nn.init.xavier_normal_(self.w_in)
        nn.init.xavier_normal_(self.w_out)
        nn.init.xavier_normal_(self.rel_basis)
        nn.init.xavier_normal_(self.rel_wt)
        nn.init.xavier_normal_(self.w_rel)
        nn.init.xavier_normal_(self.loop_rel)
        
    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
        rel_embed = torch.mm(self.rel_wt, self.rel_basis)
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_edges = edge_index.size(1) // 2
        num_ent   = x.size(0)

        if not self.cache or self.in_norm == None:
            self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
            self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

            self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(edge_index.device)
            self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(edge_index.device)

            self.in_norm     = self.compute_norm(self.in_index,  num_ent)
            self.out_norm    = self.compute_norm(self.out_index, num_ent)
        
        in_res = self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

        if self.bias: out = out + self.bias
        # if self.b_norm: out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]
    
    def com_mult(self, a, b):
        r1, i1 = a[..., 0], a[..., 1]
        r2, i2 = b[..., 0], b[..., 1]
        return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)
    
    def conj(self, a):
        a[..., 1] = -a[..., 1]
        return a
    
    def ccorr(self, a, b):
        return torch.irfft(self.com_mult(self.conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
    
    def rel_transform(self, ent_embed, rel_embed):
        if   self.opn == 'corr': 	trans_embed  = self.ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight 	= getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel  = self.rel_transform(x_j, rel_emb)
        out	= torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
        deg_inv = deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, num_rels={})'.format(
            self.__class__.__name__, self.hidden_dim, self.num_rels)