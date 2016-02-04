import common
import sys

f = open(sys.argv[1])
common.generate_dataset(f)