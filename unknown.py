# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import defaultdict
import argparse
import os
import sys


# PYTHON PROJECT IMPORTS
from trees import Tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="data file of trees to parse (one tree per line)")
    parser.add_argument("-o", "--outfile", default=sys.stdout, help="the file to write the output to")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise ValueError(f"[ERROR]: could not find file [{args.file}]")

    count: Mapping[str, int] = defaultdict(int)

    ts: Sequence[Tree] = []
    for line in open(args.infile, "r", encoding="utf8"):
        t: Tree = Tree.from_str(line)
        for leaf in t.leaves():
            count[leaf.label] += 1
        ts.append(t)

    outfile = args.outfile
    if isinstance(args.outfile, str):
        outfile = open(args.outfile, "w")

    for t in ts:
        for leaf in t.leaves():
            if count[leaf.label] < 2:
                leaf.label = "<unk>"
        outfile.write("{0}\n".format(t))

    if isinstance(args.outfile, str):
        outfile.close()


if __name__ == "__main__":
    main()

