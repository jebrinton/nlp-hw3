# SYSTEM IMPORTS
import os
import subprocess
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
_src_dir_ = os.path.join(_cd_, "src")
_dirs_to_add_ = [_cd_, _src_dir_]
for _dir_ in _dirs_to_add_:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _dirs_to_add_
del _src_dir_
del _cd_


# PYTHON PROJECT IMPORTS
import hw3


def pipeline(model_class, train_path, eval_source_path, eval_out_path, eval_annotation_path):
    cd = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(cd, "generated")

    # init and train the model
    model = model_class()
    model.train_from_file(train_path)
    model.finalize(model.get_start())

    # generate the dev.parses file
    dev_data = hw3.loaders.load_file_text(eval_source_path)

    with open(eval_out_path, "w") as f:
        for line in dev_data:
            parse, logprob = model.cky_zackterbi(line)
            f.write("%s\n" % parse)

    # do postprocessing
    postprocess_and_eval_script_path = os.path.join(cd, "postprocess_and_eval")
    subprocess.call([postprocess_and_eval_script_path, eval_out_path,
                     eval_out_path + ".post", eval_annotation_path])
    print("")


def mod_1(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod1: add unknowns for <= k count rules")
    pipeline(hw3.dtypes.AddUnkPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def mod_2(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod2: fallback vertical markov pcfg")
    pipeline(hw3.dtypes.FBVertMarkovPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def mod_3(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod3: smoothed vertical markov pcfg")
    pipeline(hw3.dtypes.SmoothVertMarkovPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def mod_4(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod4: add multi unknown symbols for <= k count rules")
    pipeline(hw3.dtypes.AddMultiUnkPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def mod_5(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod5: smoothed fallback vertical markov pcfg")
    pipeline(hw3.dtypes.SmoothFBVertMarkovPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def mod_6(train_path, eval_source_path, eval_out_path, eval_annotation_path):
    print("mod6: hard em pcfg")
    pipeline(hw3.dtypes.EMPCFG, train_path, eval_source_path,
             eval_out_path, eval_annotation_path)


def part_a(data_dir, output_dir):
    train_path = os.path.join(output_dir, "train.trees.pre")
    parse_data_path = os.path.join(data_dir, "dev.strings")
    output_parse_path = os.path.join(output_dir, "dev.parses")
    annotations_path = os.path.join(data_dir, "dev.trees")
    mod_1(train_path, parse_data_path, output_parse_path, annotations_path)
    mod_2(train_path + ".unk", parse_data_path, output_parse_path, annotations_path)
    mod_3(train_path + ".unk", parse_data_path, output_parse_path, annotations_path)
    mod_4(train_path, parse_data_path, output_parse_path, annotations_path)
    mod_5(train_path + ".unk", parse_data_path, output_parse_path, annotations_path)
    mod_6(train_path + ".unk", parse_data_path, output_parse_path, annotations_path)


def part_b(data_dir, output_dir):
    print("+----------------------+")
    print("| RUNNING ON TEST DATA |")
    print("+----------------------+")

    train_path = os.path.join(output_dir, "train.trees.pre")
    parse_data_path = os.path.join(data_dir, "test.strings")
    output_parse_path = os.path.join(output_dir, "test.parses")
    annotations_path = os.path.join(data_dir, "test.trees")

    print("Fallback vertical markov model")
    pipeline(hw3.dtypes.FBVertMarkovPCFG, train_path + ".unk", parse_data_path,
             output_parse_path, annotations_path)

    print("Smoothed fallback vertical markov model")
    pipeline(hw3.dtypes.SmoothFBVertMarkovPCFG,
             train_path + ".unk",
             parse_data_path,
             output_parse_path,
             annotations_path)


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    output_dir = os.path.join(cd, "generated")

    part_a(data_dir, output_dir)
    part_b(data_dir, output_dir)


if __name__ == "__main__":
    main()

