import numpy as np
import sys
from feature_extraction import *


def create_codebook():
    if len(sys.argv) < 2:
        print"Usage: ./generate_codebook dataset_path"
        exit(1)

    print "Parsing params"
    datasetpath = sys.argv[1]
    
    nclusters = 0
    cats = get_classes(datasetpath)
    ncats = len(cats)
    print "searching for folders at " + datasetpath
    if ncats < 1:
        print"Wrong path! \n"
        exit(1)

         
    all_files = []
    all_files_labels = {}
    
    # Obs,: all_features is a dict of form: 'image-path' : 'descriptor'
    all_features = {}
    cat_label = {}

    for cat, label in zip(cats, range(ncats)):
        # path of class
        cat_path = join(datasetpath, cat)
        # name of each image file
        cat_files = get_imgfiles(cat_path)
        # extracting features
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        # appending more features
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:   
            all_files_labels[i] = label

    print "computing the visual words via k-means"

    # passing to numpy array
    all_features_array = dict2numpy(all_features)
    
    # number of features
    nfeatures = all_features_array.shape[0]
    # number of clusters
    nclusters = 50
    
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)
    print "writing the codebook in file "
    f = open(CODEBOOK_FILE, 'wb')
    savetxt(f,codebook)

if __name__ == '__main__':
	create_codebook()
