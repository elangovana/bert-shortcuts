import argparse
import csv
import logging
import sys


def convert(input_csv, source_sep, dest_sep, output_file, quote_char='"'):
    """
Formats the input to the target by changing the separator
    """

    with open(input_csv, "r") as p_file:
        reader = csv.reader(p_file, delimiter=source_sep, quotechar=quote_char)
        header_cols = next(reader)
        print(header_cols)
        source_lines = list(reader)

    with open(output_file, "w") as w_file:
        writer = csv.writer(w_file, delimiter=dest_sep, quotechar=quote_char)
        writer.writerow(header_cols)
        for line in source_lines:
            writer.writerow(line)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile_csv",
                        help="The csv file containing predictions")

    parser.add_argument("--src_csv_sep",
                        help="The csv separator for the source", default="\t")

    parser.add_argument("--dest_csv_sep",
                        help="The csv separator for the target", default=",")

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
            args.src_csv_sep,
            args.dest_csv_sep,
            args.output)


if __name__ == '__main__':
    main_run()
