# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import Tuple
import argparse
import os
import sys


# PYTHON PROJECT IMPORTS
from trees import Tree, Node


def _brackets_helper(node: Node,
                     i: int,
                     result: Mapping[Tuple[str, int, int], int]
                     ) -> int:
    i0: int = i
    if len(node.children) > 0:
        for child in node.children:
            i = _brackets_helper(child, i, result)
        j0: int = i
        if len(node.children[0].children) > 0: # don't count preterminals
            result[node.label, i0, j0] += 1
    else:
        j0 = i0 + 1
    return j0

def brackets(t: Tree) -> Mapping[Tuple[str, int, int], int]:
    result: Mapping[Tuple[str, int, int], int] = defaultdict(int)
    _brackets_helper(t.root, 0, result)
    return result

def evalb(test_file: str,
          gold_file: str
          ) -> Mapping[str, float]:

    match_count: int = 0
    test_count: int = 0
    gold_count: int = 0

    with open(test_file, "r", encoding="utf8") as tf:
        with open(gold_file, "r", encoding="utf8") as gf:

            for test_line, gold_line in zip(tf, gf):
                gold_tree: Tree = Tree.from_str(gold_line)
                gold_brackets: Mapping[Tuple[str, int, int], int] = brackets(gold_tree)
                gold_count += sum(gold_brackets.values())

                if test_line.strip() in ["0", ""]:
                    continue
                
                test_tree = Tree.from_str(test_line)
                test_brackets: Mapping[Tuple[str, int, int], int] = brackets(test_tree)
                test_count += sum(test_brackets.values())

                for bracket, count in test_brackets.items():
                    match_count += min(count, gold_brackets[bracket])

    print("%s\t%d brackets" % (test_file, test_count))
    print("%s\t%d brackets" % (gold_file, gold_count))
    print("matching\t%d brackets" % match_count)
    print("precision\t%s" % (float(match_count)/test_count))
    print("recall\t%s" % (float(match_count)/gold_count))
    print("F1\t%s" % (2./(gold_count/float(match_count) + test_count/float(match_count))))
    return {
        "precision": (float(match_count)/test_count),
        "recall": (float(match_count)/gold_count),
        "F1": (2./(gold_count/float(match_count) + test_count/float(match_count)))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=str, help="file of predicted trees")
    parser.add_argument("gold_file", type=str, help="file of ground-truth trees")
    args = parser.parse_args()

    for fp in [args.test_file, args.gold_file]:
        if not os.path.exists(fp):
            raise ValueError(f"[ERROR]: could not find file [{fp}]")

    evalb(args.test_file, args.gold_file)


if __name__ == "__main__":
    main()

