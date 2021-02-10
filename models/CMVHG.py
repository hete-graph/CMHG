import torch
import torch.nn as nn
import torch.nn.functional as F
from embedder import embedder
from layers import MultiDiscriminator, Attention, RGCN, HRGCN, AvgReadout
import numpy as np
import datetime
from evaluate import evaluate
from utils.process import dropout_adj, drop_feature, one_one_negative_sampling, \
    one_one_rel_negative_sampling, get_adj_idx, get_adj_batch


class CMVHG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        dt = datetime.datetime.now()
        date = f"{dt.year}_{dt.month}_{dt.day}_{dt.time()}"
        feature = self.features[0].to(self.args.device)
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        adj_idx = get_adj_idx(adj, self.args.dropadj_1)

        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)

        def self_train():
            b_xent = nn.BCEWithLogitsLoss()
            xent = nn.CrossEntropyLoss()
            cnt_wait = 0
            best = 1e9
            for epoch in range(self.args.nb_epochs):
                lbl_1 = torch.ones(1, self.args.nb_nodes * 2)
                lbl_2 = torch.zeros(1, self.args.nb_nodes * 2)
                lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

                # corruption structure: drop adj
                adjs_1 = [dropout_adj(a, self.args.dropadj_1) for a in adj]
                adjs_2 = [dropout_adj(a, self.args.dropadj_2) for a in adj]

                # corruption features: drop feats
                ft_1 = drop_feature(feature, self.args.dropfeat_1)
                ft_2 = drop_feature(feature, self.args.dropfeat_2)
                fts = [ft_1, ft_2]

                # negative samples: random feature permutation
                idx = np.random.permutation(self.args.nb_nodes)
                shuf_fts = feature[:, idx, :]
                shuf_fts = shuf_fts.to(self.args.device)

                # train
                batch_idxs = get_adj_batch(adj_idx, self.args.sample_size)
                model.train()
                optimiser.zero_grad()
                result = model(fts, shuf_fts, adjs_1, adjs_2, self.args.sparse, batch_idxs)

                # local-global contrastive loss
                logits = result['logits']
                xent_loss = None
                if torch.is_tensor(logits):
                    xent_loss = b_xent(logits, lbl)
                else:
                    for view_idx, logit in enumerate(logits):
                        if xent_loss is None:
                            xent_loss = b_xent(logit, lbl)
                        else:
                            xent_loss += b_xent(logit, lbl)
                loss = xent_loss

                # total loss
                loss += self.args.reg_coef * result['reg_loss'] + self.args.w_node * result['node_loss'] + self.args.w_rel * result['rel_loss']

                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(model.state_dict(),
                               'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, date))
                else:
                    cnt_wait += 1
                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()

        self_train()

        # Evaluation
        model.load_state_dict(torch.load(
            'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, date)))
        model.eval()
        embeds = model.embed(feature, adj, self.args.sparse)
        res = evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels,
                       self.args.device)
        return res


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.rgcn_conv = nn.ModuleList(
            [RGCN(args.ft_size, args.hid_units, args.activation, args.drop_prob,
                  num_rels=args.nb_graphs, num_bases=args.bases, isBias=args.isBias) for _ in
             range(args.num_layers)])
        self.hrgcn_conv = nn.ModuleList(
            [HRGCN(args.ft_size, args.hid_units, args.activation, args.drop_prob,
                   num_rels=self.args.nb_graphs, num_bases=self.args.bases, isBias=args.isBias) for _ in
             range(args.num_layers)])
        self.R = nn.Parameter(torch.FloatTensor(args.nb_graphs, args.hid_units))
        self.cls_rel = torch.nn.Linear(args.hid_units, 1)

        # lg parameters
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.local_global_disc = MultiDiscriminator(args.hid_units)

        # ll parameters
        n_proj_h = args.hid_units
        self.fc1 = torch.nn.Linear(args.hid_units, n_proj_h)
        self.fc2 = torch.nn.Linear(n_proj_h, args.hid_units)
        self.f_k = nn.ModuleList([nn.Bilinear(args.hid_units, args.hid_units, 1) for _ in range(self.args.nb_graphs)])
        self.f_k_node = nn.Bilinear(args.hid_units, args.hid_units, 1)
        for m in self.modules():
            self.weights_init(m)

        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        self.init_weight()

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def init_weight(self):
        nn.init.xavier_normal_(self.R)

    def forward(self, seq1, seq2, adj, adj_2, sparse, batchs=None):
        '''
        seq1: positive samples
        seq2: negative samples
        '''
        result = {}
        logits = []

        if len(seq1) == 2:
            seq1_1 = seq1[0]
            seq1_2 = seq1[1]
            seq2_1 = seq2
            seq2_2 = seq2
        else:
            seq1_1 = seq1
            seq1_2 = seq1
            seq2_1 = seq2
            seq2_2 = seq2

        # graph encoders
        h_pos_1 = self.rgcn_conv[0](seq1_1, adj)  # pos emb: view1
        h_pos_2 = self.hrgcn_conv[0](seq1_2, adj_2)  # pos emb: view2
        h_neg_1 = self.rgcn_conv[0](seq2_1, adj)  # neg emb: view1
        h_neg_2 = self.hrgcn_conv[0](seq2_2, adj_2)  # neg emb: view2
        for i in range(1, self.args.num_layers):
            h_pos_1 = self.rgcn_conv[i](h_pos_1, adj)  # pos emb: view1
            h_pos_2 = self.hrgcn_conv[i](h_pos_2, adj_2)  # pos emb: view2
            h_neg_1 = self.rgcn_conv[i](h_neg_1, adj)  # neg emb: view1
            h_neg_2 = self.hrgcn_conv[i](h_neg_2, adj_2)  # neg emb: view2
        c_pos_1 = self.sigm(self.read(h_pos_1))
        c_pos_2 = self.sigm(self.read(h_pos_2))

        # Attention or not
        if self.args.isAttn:
            h_pos_1_all_lst = []
            h_neg_1_all_lst = []
            h_pos_2_all_lst = []
            h_neg_2_all_lst = []
            c_1_all_lst = []
            c_2_all_lst = []

            for h_idx in range(self.args.nheads):
                h_pos_1_all_, h_neg_1_all_, c_1_all_ = self.attn[h_idx](h_pos_1, h_neg_1, c_pos_1)
                h_pos_2_all_, h_neg_2_all_, c_2_all_ = self.attn[h_idx](h_pos_2, h_neg_2, c_pos_2)
                h_pos_1_all_lst.append(h_pos_1_all_)
                h_neg_1_all_lst.append(h_neg_1_all_)
                h_pos_2_all_lst.append(h_pos_2_all_)
                h_neg_2_all_lst.append(h_neg_2_all_)
                c_1_all_lst.append(c_1_all_)
                c_2_all_lst.append(c_2_all_)

            h_pos_1_all = torch.mean(torch.cat(h_pos_1_all_lst, 0), 0).unsqueeze(0)
            h_neg_1_all = torch.mean(torch.cat(h_neg_1_all_lst, 0), 0).unsqueeze(0)
            h_pos_2_all = torch.mean(torch.cat(h_pos_2_all_lst, 0), 0).unsqueeze(0)
            h_neg_2_all = torch.mean(torch.cat(h_neg_2_all_lst, 0), 0).unsqueeze(0)
            c_1_all = torch.mean(torch.cat(c_1_all_lst, 0), 0).unsqueeze(0)
            c_2_all = torch.mean(torch.cat(c_2_all_lst, 0), 0).unsqueeze(0)
        else:
            h_pos_1_all = torch.mean(h_pos_1, 0).unsqueeze(0)
            h_pos_2_all = torch.mean(h_pos_2, 0).unsqueeze(0)
            h_neg_1_all = torch.mean(h_neg_1, 0).unsqueeze(0)
            h_neg_2_all = torch.mean(h_neg_2, 0).unsqueeze(0)
            c_1_all = torch.mean(c_pos_1, 0).unsqueeze(0)
            c_2_all = torch.mean(c_pos_2, 0).unsqueeze(0)

        # local-global graph loss
        logit = self.local_global_disc(c_1_all, c_2_all,
                                       h_pos_1_all, h_pos_2_all,
                                       h_neg_1_all, h_neg_2_all)
        logits.append(logit)

        reg_loss = 0.0
        h_pos_all = (h_pos_1_all + h_pos_2_all) * 0.5
        for i in range(self.args.nb_graphs):

            # local-global subgraph loss
            logit = self.local_global_disc(torch.unsqueeze(c_pos_1[i], 0), torch.unsqueeze(c_pos_2[i], 0),
                                           torch.unsqueeze(h_pos_1[i], 0), torch.unsqueeze(h_pos_2[i], 0),
                                           torch.unsqueeze(h_neg_1[i], 0), torch.unsqueeze(h_neg_2[i], 0))
            logits.append(logit)

            # reg loss
            h_pos = (h_pos_1[i] + h_pos_2[i]) * 0.5
            h_neg = (h_neg_1[i] + h_neg_2[i]) * 0.5
            pos_reg_loss = ((h_pos_all - h_pos) ** 2).sum()
            neg_reg_loss = ((h_pos_all - h_neg) ** 2).sum()
            reg_loss = reg_loss + pos_reg_loss - neg_reg_loss

        result['logits'] = logits
        result['reg_loss'] = reg_loss

        # local-local contrastive loss
        h_view1 = self.projection(torch.squeeze(h_pos_1_all))
        h_view2 = self.projection(torch.squeeze(h_pos_2_all))

        node_loss = self.node_cross_entropy_loss(adj, h_view1, h_view2, batchs)
        result['node_loss'] = node_loss

        rel_loss = self.rel_cross_entropy_loss(adj, h_view1, h_view2, batchs)
        result['rel_loss'] = rel_loss

        return result

    def embed(self, seq1, adj, sparse):
        h_1 = self.rgcn_conv[0](seq1, adj)  # pos emb: view1
        h_2 = self.hrgcn_conv[0](seq1, adj)  # pos emb: view2
        for i in range(1, self.args.num_layers):
            h_1 = self.rgcn_conv[i](h_1, adj)
            h_2 = self.hrgcn_conv[i](h_2, adj)

        if self.args.isAttn:
            h_1_all_lst = []
            h_2_all_lst = []
            for h_idx in range(self.args.nheads):
                h_1_all_ = self.attn[h_idx](h_1)
                h_2_all_ = self.attn[h_idx](h_2)
                h_1_all_lst.append(h_1_all_)
                h_2_all_lst.append(h_2_all_)
            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)
        else:
            h_1_all = torch.mean(h_1, 0).unsqueeze(0)
            h_2_all = torch.mean(h_2, 0).unsqueeze(0)

        h_all = (h_1_all + h_2_all) * 0.5
        return h_all.detach()

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    # node_loss
    def node_sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if self.args.sigm:
            res = self.sigm(torch.squeeze(self.f_k_node(z1, z2)))
        else:
            res = torch.squeeze(self.f_k_node(z1, z2))
        return res

    def node_bilinear_logit(self, i, j, k, z1: torch.Tensor, z2: torch.Tensor):
        k = torch.squeeze(k)

        intra_pos_logit = self.node_sim(z1[i], z1[j]).view(-1, )
        inter_pos_logit = self.node_sim(z1[i], z2[j]).view(-1, )

        intra_neg_logit = self.node_sim(z1[i], z1[k]).view(-1, )
        inter_neg_logit = self.node_sim(z1[i], z2[k]).view(-1, )

        logit = torch.cat([intra_pos_logit, inter_pos_logit, intra_neg_logit, inter_neg_logit], dim=0)

        return logit

    def node_cross_entropy_loss(self, adjs, z1: torch.Tensor, z2: torch.Tensor, batchs=None, mean: bool = True):
        i, j, k = one_one_negative_sampling(adjs, batchs, 1)
        loss = 0.0
        b_xent = nn.BCEWithLogitsLoss()
        for r in range(self.args.nb_graphs):
            lbl_1 = torch.ones(1, i[r].size(0) * 2)
            lbl_2 = torch.zeros(1, i[r].size(0) * 2)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            logit = self.node_bilinear_logit(i[r], j[r], k[r], z1, z2)
            logit_ = self.node_bilinear_logit(i[r], j[r], k[r], z2, z1)
            logit = (logit + logit_) * 0.5
            logit = torch.squeeze(logit).unsqueeze(0)
            loss += b_xent(logit, lbl)

        return loss

    # rel_loss
    def rel_sim(self, r, z1: torch.Tensor, z2: torch.Tensor):
        if self.args.sigm:
            res = self.sigm(torch.squeeze(self.f_k[r](z1, z2)))
        else:
            res = torch.squeeze(self.f_k[r](z1, z2))
        return res

    def rel_bilinear_logit(self, r, num_rel, i, j, rs, z1, z2):
        intra_pos_logit = self.rel_sim(r, z1[i], z1[j]).view(-1, )
        inter_pos_logit = self.rel_sim(r, z1[i], z2[j]).view(-1, )

        neg_logit = []
        for _r in range(num_rel):
            if _r == r:
                continue
            mask_r = torch.eq(rs, _r)
            rest = mask_r.nonzero(as_tuple=False).view(-1)
            if rest.numel() > 0:
                intra_neg_logit = self.rel_sim(_r, z1[i[mask_r]], z1[j[mask_r]]).view(-1, )
                inter_neg_logit = self.rel_sim(_r, z1[i[mask_r]], z2[j[mask_r]]).view(-1, )
                neg_logit.append(intra_neg_logit)
                neg_logit.append(inter_neg_logit)

        neg_logit = torch.cat(neg_logit, dim=0)
        logit = torch.cat([intra_pos_logit, inter_pos_logit, neg_logit], dim=0)

        return logit

    def rel_cross_entropy_loss(self, adjs, z1: torch.Tensor, z2: torch.Tensor, batchs, mean: bool = True):
        i, j, rs = one_one_rel_negative_sampling(adjs, batchs)
        loss = 0.0
        b_xent = nn.BCEWithLogitsLoss()
        for r in range(self.args.nb_graphs):
            if len(i[r]) == 0:
                continue

            lbl_1 = torch.ones(1, i[r].size(0) * 2)
            lbl_2 = torch.zeros(1, i[r].size(0) * 2)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            logit = self.rel_bilinear_logit(r, self.args.nb_graphs, i[r], j[r], rs[r], z1, z2)
            logit_ = self.rel_bilinear_logit(r, self.args.nb_graphs, i[r], j[r], rs[r], z2, z1)
            logit = (logit + logit_) * 0.5
            logit = torch.squeeze(logit).unsqueeze(0)
            loss += b_xent(logit, lbl)

        return loss
