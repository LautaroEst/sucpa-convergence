
import argparse

from sucpa import SUCPA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--steps', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load logits and labels
    # ...



if __name__ == '__main__':
    main()