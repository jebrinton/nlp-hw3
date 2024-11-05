# SYSTEM IMPORTS
from collections.abc import Mapping
from typing import Tuple
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from models.parser import Parser


def build_dict(grammar_string: str) -> Mapping[str, float]:
    rules_to_vals: Mapping[str, float] = dict()
    for line in grammar_string.split("\n"):
        if "#" not in line:
            continue
        rule, num_str = line.split("#")
        rules_to_vals[rule] = float(num_str)
    return rules_to_vals


def part_a(grammar_string_counts: str,
           max_num: int = 5
           ) -> None:
    rules_dict: Mapping[str, float] = build_dict(grammar_string_counts)

    # sort by count
    rules_list: Sequence[Tuple[str, float]] = sorted(rules_dict.items(), key=lambda x: x[1], reverse=True)

    assert(len(rules_list) == len(set(rules_dict.keys())))  # assert all rules are unique
    print("number of unique rules: %s" % len(rules_list))
    print("top %s rules and counts:" % max_num)

    for rule, count in rules_list[:max_num]:
        print("\trule [{0}] occured [{1}]".format(rule.strip(), int(count)))


def part_b(grammar_string_probs: str,
           max_num: int = 5
           ) -> None:
    rules_dict: Mapping[str, float] = build_dict(grammar_string_probs)

    # sort by prob
    rules_list: Sequence[Tuple[str, float]] = sorted(rules_dict.items(), key=lambda x: x[1], reverse=True)

    assert(len(rules_list) == len(set(rules_dict.keys())))  # assert all rules are unique
    print("grammar:\n%s" % grammar_string_probs)
    print("top %s rules and probs:" % max_num)

    for rule, count in rules_list[:max_num]:
        print("\trule [{0}] has prob [{1:.3f}]".format(rule.strip(), count))


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cd, "data", "train.trees")
    out_path = os.path.join(cd, "generated", "train.trees.pre.unk")

    model = Parser()
    model.train_from_file(out_path, already_cnf=True)
    start_nonterm = "TOP"
    model.grammar.set_start(start_nonterm)
    raw_counts_rules = "%s" % model

    model.finalize(start_nonterm)
    cond_prob_rules = "%s" % model
    max_num = 5

    part_a(raw_counts_rules, max_num=max_num)
    part_b(cond_prob_rules, max_num=max_num)


if __name__ == "__main__":
    main()

