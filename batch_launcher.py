from aostar import ao_star
import argparse
import logging
import datetime

if __name__ != "__main__":
    raise RuntimeError("This file is not meant to be imported as a library")

batch_launch_time_str = datetime.datetime.now().strftime("%Y %b-%d %H:%M:%S")
parser = argparse.ArgumentParser()
parser.add_argument('lean_file', type=str, help="The lean file that contains theorems to prove (All to-be-proven theorems should be provisionally proved as `begin\\n  sorry\\nend`.))")
args = parser.parse_args()
lean_file = args.lean_file
global_logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, filemode="w")