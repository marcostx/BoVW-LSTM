import feature_extraction
from sklearn.feature_extraction import FeatureHasher
import common
import sys
import numpy as np

if __name__ == '__main__':
	X,y = common.generate_ucf_dataset('frames')