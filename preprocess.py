# SYSTEM IMPORTS
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

    outfile = args.outfile
    if isinstance(args.outfile, str):
        outfile = open(args.outfile, "w")


    for lidx, line in enumerate(open(args.infile, "r", encoding="utf8")):

        t: Tree = Tree.from_str(line)

        # Binarize, inserting 'X*' nodes.
        t.binarize()

        # Remove unary nodes
        t.remove_unit()

        # The tree is now strictly binary branching, so that the CFG is in Chomsky normal form.

        # Make sure that all the roots still have the same label.
        assert t.root.label == 'TOP', f'line {lidx}: {t.root.label}'

        outfile.write("%s\n" % t)

    if isinstance(args.outfile, str):
        outfile.close()


if __name__ == "__main__":
    main()

