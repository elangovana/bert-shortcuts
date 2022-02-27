import argparse
import csv
import logging
import sys


def convert(input_csv, source_sep, dest_sep, output_file, label_col, label_order, quote_char='"'):
    """
Formats the input to the target by changing the separator
    """
    label_map = {l: f"{i:03}_{l}" for i, l in enumerate(label_order)}
    print(label_map)

    with open(input_csv, "r") as p_file:
        reader = csv.reader(p_file, delimiter=source_sep, quotechar=quote_char)
        header_cols = next(reader)
        print(header_cols)
        source_lines = list(reader)

    label_index = list(filter(lambda x: x[1] == label_col, enumerate(header_cols)))[0][0]
    print(label_index)

    with open(output_file, "w") as w_file:
        writer = csv.writer(w_file, delimiter=dest_sep, quotechar=quote_char)
        header_cols[label_index] = "label"
        writer.writerow(header_cols)
        for line in source_lines:
            line[label_index] = label_map[line[label_index]]
            writer.writerow(line)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datafile_csv",
                        help="The csv file containing predictions")

    parser.add_argument("label_col",
                        help="The name of the label column")

    parser.add_argument("labels_in_order_csv",
                        help="The label names in order (csv) of index in the model. This is work around the default_glue_script")

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
            args.output,
            args.label_col,
            args.labels_in_order_csv.split(",")
            )


if __name__ == '__main__':
    main_run()
