# SYSTEM IMPORTS
from typing import Sequence, Union
import argparse
import os
import sys


# PYTHON PROJECT IMPORTS
from trees import Tree


def postprocess(in_file: str,
                out_file: Union[str, object]) -> None:
    out_trees: Sequence[str] = list()

    if isinstance(out_file, str):
        out_file = open(out_file, "w")

    with open(in_file, "r", encoding="utf8") as f:
        for tree_string in f:
            try:
                t = Tree.from_str(tree_string)

                t.restore_unit()
                t.unbinarize()

                out_file.write("%s\n" % t)
            except Exception as e:
                # print("AAA", e)
                out_file.write("\n")

    if not isinstance(out_file, str):
        out_file.close()

    return out_trees


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="data file of trees to parse (one tree per line)")
    parser.add_argument("-o", "--outfile", default=sys.stdout, help="the file to write the output to")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise ValueError(f"[ERROR]: could not find file [{args.file}]")

    postprocess(args.infile, args.outfile)


if __name__ == "__main__":
    main()
    
