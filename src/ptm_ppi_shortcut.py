import argparse
import logging
import sys

import pandas as pd
import sklearn

from model_nb_tree_classifier import ModelNBTreeClassifier


def train(trainfile, testfile=None):
    df_train = pd.read_json(trainfile, orient="records")
    m = ModelNBTreeClassifier("PROTPART1", "PROTPART0")
    m.train(df_train["x"], df_train["y"])
    if testfile is not None:
        df_test = pd.read_json(testfile, orient="records")
        actual = m.predict(df_test["x"])
        pos_f1 = sklearn.metrics.f1_score(df_test["y"], actual, labels=[1, 2, 3, 4, 5, 6], average='micro',
                                          sample_weight=None, zero_division='warn')
        all_f1 = sklearn.metrics.f1_score(df_test["y"], actual, average='micro', sample_weight=None, zero_division='warn')
        print(sklearn.metrics.classification_report(df_test["y"],
                                                    actual,
                                                    output_dict=False,
                                                    labels=[1, 2, 3, 4, 5, 6]))

        # print(sklearn.metrics.classification_report(df_test["y"],
        #                                             actual,
        #                                             output_dict=False))
        print("Pos labels", pos_f1, "All labels", all_f1)

    return m


def run_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--trainfile",
                        help="The input ppi multiclass train file", required=True)

    parser.add_argument("--testfile",
                        help="The input ppi multiclass test file", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print(args.__dict__)

    train(args.trainfile, args.testfile)




if __name__ == "__main__":
    run_main()
