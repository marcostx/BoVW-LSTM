from common import *
import sys

if len(sys.argv) < 2:
	raise("Missing params! ")

# taking the video frames
process_ucf_dataset(sys.argv[1])