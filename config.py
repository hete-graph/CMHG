import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CMVHG')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--embedder', default='CMVHG')
    parser.add_argument('--dataset', default='acm')
    parser.add_argument('--metapaths', default='')
    parser.add_argument('--hid_units', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bases', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--drop_prob', type=float, default=0.5)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--w_node', type=float, default=0)
    parser.add_argument('--w_rel', type=float, default=0)
    parser.add_argument('--dropadj_1', type=float, default=0.1)
    parser.add_argument('--dropadj_2', type=float, default=0.2)
    parser.add_argument('--dropfeat_1', type=float, default=0.1)
    parser.add_argument('--dropfeat_2', type=float, default=0.1)
    parser.add_argument('--reg_coef', type=float, default=0.01, help='consensus reg')
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=False)

    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--sigm', action='store_true', default=False)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--head', type=int, default=8, help='GAT head')
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', default='relu')

    return parser.parse_known_args()
