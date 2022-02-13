import argparse
import logging
import os
import sys
import tarfile
import shutil


def model_untar_and_package(model_tar_gz_path, config_file_path, vocab_file_path, output_dir):
    """
    Packages the model and the config file with vocab into the output dir
    """
    logger = logging.getLogger(__name__)
    with  tarfile.open(model_tar_gz_path) as f:
        f.extractall(output_dir)

    shutil.copyfile(config_file_path, os.path.join(output_dir, os.path.basename(config_file_path)))

    shutil.copyfile(vocab_file_path, os.path.join(output_dir, os.path.basename(vocab_file_path)))
    logger.info("Files in output directory: {}".format(os.listdir(output_dir)))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--modeltarfile",
                        help="The model.tar.gz.file", required=True)

    parser.add_argument("--modelconfigfile",
                        help="The model_config file",
                        required=True)

    parser.add_argument("--vocabfile",
                        help="The vocab file",
                        required=True)

    parser.add_argument("--outdir", help="The output dir", required=True)

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
    model_untar_and_package(args.modeltarfile,
                            args.modelconfigfile,
                            args.vocabfile,
                            args.outdir
                            )


if __name__ == '__main__':
    main_run()
