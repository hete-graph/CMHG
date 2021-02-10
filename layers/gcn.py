import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from geoopt import PoincareBall
from geoopt.manifolds.stereographic import math as pmath
from torch_geometric.nn.inits import glorot


class RGCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, num_rels=1, num_bases=1, isBias=False):
        super(RGCN, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.comp = nn.Parameter(torch.Tensor(num_rels, self.num_bases))
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, in_ft, out_ft))

        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        if isBias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.drop_prob = drop_prob
        self.isBias = isBias
        self.weights_init()

    def weights_init(self):
        glorot(self.weight)
        glorot(self.comp)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seqs, adjs):
        out = []
        weight = torch.mm(self.comp, self.weight.view(self.num_bases, -1)).view(
            self.num_rels, self.in_ft, self.out_ft)
        seq = F.dropout(seqs, self.drop_prob, training=self.training)

        for i in range(self.num_rels):
            # seq = seqs[i]
            adj = adjs[i]

            h = torch.mm(torch.squeeze(seq), weight[i])
            h = torch.unsqueeze(torch.spmm(adj, torch.squeeze(h, 0)), 0)

            # seq += torch.unsqueeze(self.bias[i], dim=0)
            out.append(h)

        out = torch.squeeze(torch.stack(out))

        if self.isBias:
            out += torch.unsqueeze(self.bias, dim=0)
        return self.act(out)


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, ball, in_features, out_features, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.ball = ball
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, xs):
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.ball.mobius_matvec(weight, xs)
        res = pmath.project(mv, k=torch.tensor(1.0))
        # bias = self.manifold.proj_tan0(self.bias, self.c)
        hyp_bias = self.ball.expmap0(self.bias)
        hyp_bias = pmath.project(hyp_bias, k=torch.tensor(1.0))
        res = self.ball.mobius_add(res, hyp_bias)
        res = pmath.project(res, k=torch.tensor(1.0))
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, ball, in_features, dropout):
        super(HypAgg, self).__init__()
        self.ball = ball
        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        if adj.ndim == 3:
            support_t = torch.stack([torch.spmm(adj[i], self.ball.logmap0(x[i])) for i in range(adj.size(0))])
        else:
            x_tangent = self.ball.logmap0(x)
            support_t = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x_tangent, 0)), 0)

        output = pmath.project(self.ball.expmap0(support_t), k=torch.tensor(1.0))
        return output


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, ball, act):
        super(HypAct, self).__init__()
        self.ball = ball
        self.act = act

    def forward(self, x):
        xt = self.act(self.ball.logmap0(x))
        # xt = pmath.project(xt, k=torch.tensor(1.0))
        return pmath.project(self.ball.expmap0(xt), k=torch.tensor(1.0))


class HRGCN(nn.Module):
    """
    Hyperbolic relational graph convolution layer.
    """

    def __init__(self, in_ft, out_ft, act, dropout, num_rels=1, num_bases=1, isBias=True):
        super(HRGCN, self).__init__()
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        self.drop_prob = dropout
        self.ball = PoincareBall(c=1.0)
        # self.linear = HypLinear(self.ball, in_ft, out_ft, dropout, bias, num_rels, num_bases)
        self.agg = HypAgg(self.ball, out_ft, dropout)
        self.hyp_act = HypAct(self.ball, self.act)

        self.in_ft = in_ft
        self.out_ft = out_ft
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.comp = nn.Parameter(torch.Tensor(num_rels, num_bases))
        self.weight = nn.Parameter(torch.Tensor(num_bases, in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(num_rels, out_ft))
        self.bias.data.fill_(0.0)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.xavier_uniform_(self.comp, gain=math.sqrt(2))
        # glorot(self.weight)
        # glorot(self.comp)

    def forward(self, seqs, adjs):
        out = []
        weight = torch.mm(self.comp, self.weight.view(self.num_bases, -1)).view(
            self.num_rels, self.out_ft, self.in_ft)
        seq = F.dropout(seqs, self.drop_prob, training=self.training)

        for i in range(self.num_rels):
            # x = seqs[i]
            adj = adjs[i]

            # x = F.dropout(x, self.drop_prob, training=self.training)
            h = self.ball.expmap0(seq)

            # h = self.linear.forward(x)
            h = self.ball.mobius_matvec(weight[i], h)
            h = pmath.project(h, k=torch.tensor(1.0))

            # bias = pmath.project(self.bias, k=torch.tensor(1.0))
            hyp_bias = self.ball.expmap0(self.bias[i])
            hyp_bias = pmath.project(hyp_bias, k=torch.tensor(1.0))
            h = self.ball.mobius_add(h, hyp_bias)
            h = pmath.project(h, k=torch.tensor(1.0))

            h = self.agg.forward(h, adj)
            h = self.hyp_act.forward(h)
            h = self.ball.logmap0(h)  # transfer hyperbolic features to Euclidean
            out.append(h)

        out = torch.squeeze(torch.stack(out))
        # out = torch.squeeze(torch.mean(out, dim=0))
        return out
