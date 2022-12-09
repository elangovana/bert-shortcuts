import argparse
import logging
import os
import sys
import tarfile
import shutil

from transformers import BertForSequenceClassification


def model_untar_and_package_bert_sequence(model_tar_gz_path, config_file_path, vocab_file_path, output_dir):
    """
    Packages the model and the config file with vocab into the output dir
    """
    logger = logging.getLogger(__name__)
    with  tarfile.open(model_tar_gz_path) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output_dir)

    save_base_bert_sequence(output_dir)

    shutil.copyfile(config_file_path, os.path.join(output_dir, os.path.basename(config_file_path)))

    shutil.copyfile(vocab_file_path, os.path.join(output_dir, os.path.basename(vocab_file_path)))

    logger.info("Files extracted in output directory: {}".format(os.listdir(output_dir)))


def save_base_bert_sequence(model_dir):
    logger = logging.getLogger(__name__)
    logger.info("Saving just the base bert without classifier")
    BertForSequenceClassification.from_pretrained(model_dir).bert.save_pretrained(model_dir)


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
    model_untar_and_package_bert_sequence(args.modeltarfile,
                            args.modelconfigfile,
                            args.vocabfile,
                            args.outdir
                            )


if __name__ == '__main__':
    main_run()
