from config import parse_args
from main import main

if __name__ == '__main__':
    args, unknown = parse_args()
    res = main(args)
