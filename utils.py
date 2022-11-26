import logging
import sys
import os


def log_config(file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging