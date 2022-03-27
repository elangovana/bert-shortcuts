import argparse
import csv
import logging
import sys

from src.utils.predict_entailment import PredictEntailment


class PredictMnliEntailment:
    """
    Entailment task predictor for MNLI Tsv dataset
    """

    def __init__(self, prediction_csv_file, outputfile, model_or_path, tokenisor_or_path,
                 do_lower_case=False, max_length=512, batch_size=8):
        sentence_a = []
        sentence_b = []
        labels = []
        with open(prediction_csv_file, "r") as f:
            for r in csv.DictReader(f):
                sentence_a.append(r["sentence1"])
                sentence_b.append(r["sentence2"])
                labels.append(r["label"])

        label_names_in_order = sorted(list(set(labels)))
        self._logger.info(f"Using label names in order {label_names_in_order}")

        predictor = PredictEntailment(model_or_path, tokenisor_or_path,
                                      do_lower_case, max_length, batch_size)

        predictor(sentence_a, sentence_b, labels,
                  lambda a, b, c: predictor.write_to_file(outputfile, a, b, c, label_names_in_order)
                  )

    @property
    def _logger(self):
        return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_csv",
                        help="The csv file containing data to predict")

    parser.add_argument("output_csv",
                        help="The output csv file")

    parser.add_argument("model_path",
                        help="The model path")

    parser.add_argument("tokenisor_path",
                        help="The tokenisor path")

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    return args


def main_run():
    args = parse_args()
    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Runs
    PredictMnliEntailment(args.input_csv,
                          args.src_csv_sep,
                          args.output_csv,
                          args.model_path,
                          args.tokenisor_path
                          )


if __name__ == '__main__':
    main_run()
