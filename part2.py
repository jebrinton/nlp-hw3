# SYSTEM IMPORTS
from collections.abc import Mapping
from scipy.optimize import curve_fit
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import timeit


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from models.parser import Parser
from eval.evalb import evalb
from postprocess import postprocess


def part_a(model: Parser,
           output_dir: str) -> None:
    cd = os.path.abspath(os.path.dirname(__file__))

    # path to the dev data
    data_path = os.path.join(cd, "data", "dev.strings")

    # path to where we need to write the output parses
    output_path = os.path.join(output_dir, "dev.parses")

    # load in each line from the dev file
    dev_data: Sequence[str] = list()
    with open(data_path, "r", encoding="utf8") as f:
        dev_data = f.readlines()

    # find the best parse for each line. Write the best parse to a separate line in the output path
    # also print to the console the best parse for the first k lines
    num_parses_to_show = 5
    with open(output_path, "w") as f:
        for line in dev_data:
            print(line)
            # parse, logprob = model.cky_viterbi(line) <- this is the old line
            parse, logprob = model.cky_viterbi(line)

            # parse can fail
            if parse is None:
                parse = ""

            # print the parse if its in the first k lines
            if num_parses_to_show > 0:
                print("parse: [{0}], logprob: [{1:.3f}]".format(parse, logprob))
                num_parses_to_show -= 1

            # write the parse to the file
            f.write("%s\n" % parse)


def part_b(model: Parser) -> None:
    # As stated in the pdf, this function should re-evaluate your parser on the dev set
    # but this time record how long a valid parse takes to run (I would recommend using the timeit module
    # that I have already imported). You should generate a log-log plot from this data, where the x axis
    # is the sentence length (log-scale) and the y-axis is the runtime (log-scale). Estimate the value of
    # k for which y = cx^k (I would recommend doing a least-squares fit). Is your value of k close to
    # 3? Why or why not?
    return


def part_c(model: Parser,
           output_dir: str) -> Mapping[str, float]:
    cd = os.path.abspath(os.path.dirname(__file__))

    dev_parse_file = os.path.join(output_dir, "dev.parses")
    dev_parse_postprocess_file = os.path.join(output_dir, "dev.parses.post")

    data_dir = os.path.join(cd, "data")
    gold_file = os.path.join(data_dir, "dev.trees")

    # post process the dev parses
    postprocess(dev_parse_file, dev_parse_postprocess_file)

    # eval against ground truth
    return evalb(dev_parse_postprocess_file, gold_file)


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cd, "data", "train.trees")
    output_dir = os.path.join(cd, "generated")
    out_path = os.path.join(output_dir, "train.trees.pre.unk")

    model = Parser()
    model.train_from_file(out_path, already_cnf=True)
    start_nonterm = "TOP"
    model.finalize(start_nonterm)

    part_a(model, output_dir)
    # part_b(model)
    # part_c(model, output_dir)


if __name__ == "__main__":
    main()

