import torch
from torch import nn
from models.encoder import DescriptorEncoder, KeypointEncoder
#from ..models.encoder import DescriptorEncoder, KeypointEncoder

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)


    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def MLP(channels: list, do_bn=False, do_in=True, dropout=False):
    """ Multi-layer perceptron """
    assert not (do_bn and do_in)
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if do_in:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

def Conv2d(channels: list, do_bn=False, do_in=True, dropout=False, kernels=None):
    """ Multi-layer perceptron """
    assert not (do_bn and do_in)
    n = len(channels)
    layers = []
    for i in range(1, n):
        if kernels is None:
            kernel_size = 3
        else:
            kernel_size = kernels[i-1]
        layers.append(
            nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernel_size, stride=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm2d(channels[i]))
            if do_in:
                layers.append(nn.InstanceNorm2d(channels[i]))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

#I disagreed with how this was originally written
#so I re-wrote it
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads #dk and dv
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) #h*dvxdmodel
        #self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.projQ = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        self.projK = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        self.projV = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdv

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        queries = [q(query) for q in self.projQ]
        keys = [k(key) for k in self.projK]
        values = [v(value) for v in self.projV]

        query = torch.stack(queries, dim=2)
        key = torch.stack(keys, dim=2)
        value = torch.stack(values, dim=2)

        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, num_heads)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, descs_0, descs_1):
        num_batch = len(descs_0)
        num_nodes_0 = descs_0[0].shape[0]
        num_nodes_1 = descs_1[0].shape[0]

        descs_0 = torch.stack(descs_0, dim=0)
        descs_0 = torch.permute(descs_0, (0, 2, 1))

        descs_1 = torch.stack(descs_1, dim=0)
        descs_1 = torch.permute(descs_1, (0, 2, 1))

        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src_0 = descs_1
                src_1 = descs_0
            else:
                src_0 = descs_0
                src_1 = descs_1

            delta_0 = layer(descs_0, src_0)
            delta_1 = layer(descs_1, src_1)

            descs_0 = (descs_0 + delta_0)
            descs_1 = (descs_1 + delta_1)

        descs_0 = torch.permute(descs_0, (0, 2, 1))
        descs_1 = torch.permute(descs_1, (0, 2, 1))
        
        descs_0_out = [None]*num_batch
        descs_1_out = [None]*num_batch
        for i in range(num_batch):
            descs_0_out[i] = descs_0[i]
            descs_1_out[i] = descs_1[i]

        return descs_0_out, descs_1_out
        
class FruitletAssociator(nn.Module):

    default_config = {
        'keypoint_dim': 4,
        'keypoint_pool_size': 64,
        'descriptor_dim_in': 256,
        'descriptor_pool_size': 7,
        'descriptor_enc_dim': 510,
        'descriptor_dim': 512,
        'num_heads': 8,
        'GNN_layers': ['self', 'cross'] * 6,
        'sinkhorn_iterations': 100,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.denc = DescriptorEncoder()
        self.kenc = KeypointEncoder()

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'], self.config['num_heads'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data):
        #[nodesxdescriptor_dim_inxdescriptor_pool_sizexdescriptor_pool_size]xbatch_siaze
        descs_0, descs_1 = data['descriptors']

        #[nodesxdescriptor_dim_inxdescriptor_pool_sizexdescriptor_pool_size]xbatch_size
        kpts_0, kpts_1 = data['keypoints']

        is_tag_0, is_tag_1 = data['is_tag']

        assoc_scores_0, assoc_scores_1 = data['assoc_scores']

        num_batch = len(descs_0)
        num_nodes_0 = descs_0[0].shape[0]
        num_nodes_1 = descs_1[0].shape[0]

        #put in batches to speed up.  Use batch dim instead of last dim to keep instance norm consistant
        descs_0_cat = torch.cat(descs_0, dim=0)
        kpts_0_cat = torch.cat(kpts_0, dim=0)
        is_tag_0_cat = torch.cat(is_tag_0, dim=0)
        assoc_scores_0_cat = torch.cat(assoc_scores_0, dim=0)

        descs_1_cat = torch.cat(descs_1, dim=0)
        kpts_1_cat = torch.cat(kpts_1, dim=0)
        is_tag_1_cat = torch.cat(is_tag_1, dim=0)
        assoc_scores_1_cat = torch.cat(assoc_scores_1, dim=0)
        
        num_cat_0 = num_batch*num_nodes_0
        assert descs_0_cat.shape[0] == (num_cat_0)
        assert kpts_0_cat.shape[0] == (num_cat_0)

        num_cat_1 = num_batch*num_nodes_1
        assert descs_1_cat.shape[0] == (num_cat_1)
        assert kpts_1_cat.shape[0] == (num_cat_1)

        num_cat = num_cat_0 + num_cat_1
        descs_cat = torch.cat([descs_0_cat, descs_1_cat], dim=0)
        kpts_cat = torch.cat([kpts_0_cat, kpts_1_cat], dim=0)
        is_tag_cat = torch.cat([is_tag_0_cat, is_tag_1_cat], dim=0)
        assoc_scores_cat = torch.cat([assoc_scores_0_cat, assoc_scores_1_cat], dim=0)

        descs_cat = self.denc(descs_cat)
        kpts_cat = self.kenc(kpts_cat)

        descs_cat = descs_cat + kpts_cat
        descs_cat = torch.reshape(descs_cat, (num_cat, self.config['descriptor_enc_dim']))

        tag_feature = torch.zeros_like(descs_cat[:, 0:1])
        tag_feature[is_tag_cat == 1] = 1

        descs_cat = torch.cat([descs_cat, tag_feature, assoc_scores_cat.unsqueeze(-1)], dim=1)

        descs_0_cat = descs_cat[0:num_cat_0]
        descs_1_cat = descs_cat[num_cat_0:]

        descs_0 = torch.split(descs_0_cat, num_nodes_0, dim=0)
        descs_1 = torch.split(descs_1_cat, num_nodes_1, dim=0)
        
        #Multi-layer Transformer network.
        descs_0, descs_1 = self.gnn(descs_0, descs_1)

        descs_0_cat = torch.cat(descs_0, dim=0)
        descs_1_cat = torch.cat(descs_1, dim=0)
        descs_cat = torch.cat([descs_0_cat, descs_1_cat], dim=0)
        descs_cat = torch.reshape(descs_cat, (num_cat, self.config['descriptor_dim'], 1))

        mdescs = self.final_proj(descs_cat)

        mdescs_0 = mdescs[0:num_cat_0]
        mdescs_1 = mdescs[num_cat_0:]

        #TODO will this work? or should split and cat?
        mdescs_0 = torch.reshape(mdescs_0, (num_batch, num_nodes_0, self.config['descriptor_dim']))
        mdescs_1 = torch.reshape(mdescs_1, (num_batch, num_nodes_1, self.config['descriptor_dim']))

        mdescs_0 = torch.permute(mdescs_0, (0, 2, 1))
        mdescs_1 = torch.permute(mdescs_1, (0, 2, 1))
        
        # Run the optimal transport.
        scores = torch.einsum('bdn,bdm->bnm', mdescs_0, mdescs_1)
        scores = scores / self.config['descriptor_dim']**.5

        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        losses = []
        if data['return_losses']:
            M = data['M']
            for i in range(len(M)):
                xs = M[i][:, 0]
                ys = M[i][:, 1]

                #TODO sum or mean
                loss = torch.mean(-torch.log(scores[i, xs, ys].exp()))
                loss = torch.reshape(loss, (1, -1))
                losses.append(loss)

            losses = torch.cat(losses, dim=0)
        
        if data['return_scores']:
            scores_to_return = scores
        else:
            scores_to_return = None

        return {'scores': scores_to_return, 
                'losses': losses
        }
