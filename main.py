import argparse

import pandas as ps
from sklearn.ensemble import RandomForestClassifier

from tools.processing.model import Model
from tools.processing.visualize import Visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="csv file")
    parser.add_argument("-o", "--output", help="output model into file", metavar="FILE")
    parser.add_argument("-r", "--random", help="random seed value", metavar="SEED", type=int)
    args = parser.parse_args()

    data = ps.read_csv(args.file)

    model = Model(data, RandomForestClassifier(20, "entropy", 8, random_state=args.random), args.random)

    model.process(3)

    for entry in model.results.values():
        Visualize(entry['model'], entry['data'][0], entry['data'][1], entry['data'][2]).view_report()


if __name__ == "__main__":
    main()
