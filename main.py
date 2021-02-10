from models import CMVHG
from utils.process import init_seeds

Models = {
    'CMVHG': CMVHG
}


def main(args):
    if args.dataset == 'acm':
        args.metapaths = 'PAP,PLP'
    elif args.dataset == 'dblp':
        args.metapaths = 'PAP,PPrefP,PATAP'
    elif args.dataset == 'imdb':
        args.metapaths = 'MAM,MDM'
    elif args.dataset == 'amazon':
        args.metapaths = 'IVI,IBI,ITI,IOI'

    init_seeds(args.seed)
    embedder = Models[args.embedder](args)
    res = embedder.training()
    return res
