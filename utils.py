import logging
import os

from datetime import datetime

logger = logging.getLogger(__name__)


def create_dir(path_prefix, dir_name):
    output_dir = os.path.join(path_prefix, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_timestamp():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
