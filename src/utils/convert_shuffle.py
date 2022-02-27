import argparse
import csv
import logging
import random
import sys


def convert(input_csv, sep, columns_to_shuffle, output_file, quotechar='"'):
    """
Shuffles the words in sentences in the columns.
    """
    print(columns_to_shuffle)

    with open(input_csv, "r") as p_file:
        reader = csv.reader(p_file, delimiter=sep, quotechar=quotechar)
        header_cols = next(reader)
        print(header_cols)
        source_lines = list(reader)

    with open(output_file, "w") as w_file:
        writer = csv.writer(w_file, delimiter=sep, quotechar=quotechar)
        writer.writerow(header_cols)
        for line in source_lines:
            # Shuffle words in columns
            for col_i, _ in filter(lambda x: x[1] in columns_to_shuffle, enumerate(header_cols)):
                line[col_i] = _shuffle_words_in_sentence(line[col_i])
            writer.writerow(line)


def _shuffle_words_in_sentence(s):
    s = s.split(" ")
    random.shuffle(s)
    s = " ".join(s)
    return s


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile_csv",
                        help="The csv file containing predictions")

    parser.add_argument("columns_to_shuffle",
                        help="The columns containing the sentences to shuffle words in, e.g. column1,column2")

    parser.add_argument("--csv_sep",
                        help="The csv separator", default="\t", )

    parser.add_argument("--output",
                        help="The output file", required=True)

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
    convert(args.datafile_csv,
            args.csv_sep,
            args.columns_to_shuffle.split(","),
            args.output)


if __name__ == '__main__':
    main_run()
