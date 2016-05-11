"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                         Feature Extraction Module                        |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2015-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: feature_extraction.py
//  ================================
//  This module have the function of generate histograms of video frames based
//  in a codebook of visual words of a dataset pre-defined. The inputs are :
//
//      - path to dataset folder , where we'll compute the codebook (if it doesn't
//  exists already). This folder must have the following form:
            
            |-- path_to_dataset
            |    |-- class1
            |    |-- class2
            |    |-- class3
            ...
            |    |--- classN
            
//       where each class(i) contains a group of images belonging to class(i)
//      - path to folder of input video frames. This folder must have the following 
//  form:

            |-- path_to_folder_of_input_video
            |    |-- frame-0
            |    |-- frame-1
            |    |-- frame-2
            ...
            |    |-- frame-N 
         
         
//
//  The output are the histograms of video frames.
//
//  Supossing that we get len(codebook) == 100, in other words, we'll compute 100
//  visual words at our dataset of images and we'll generate a histogram of   
//  100 visual words for each frame of the video. The result of this program is a
//  file that has the histogram of each video frame. This file is saved into
//  the folder containing the video frames sequence.
    

// Parameters:
// sys.argv[1] => dataset path
// sys.argv[2] => input folder video frames

"""

# Libs
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, fromstring, asarray
from os.path import exists, isdir, basename, isfile, join, splitext
from sklearn.feature_extraction import FeatureHasher
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
from glob import glob
import numpy as np
import argparse
import sys

# Files / methods
from common import *
import sift

# name of file where will be saved the histograms of visual words for each frame 
HISTOGRAMS_FILE = 'histograms.lstm'

# name of the output hash matrix file
HASHMATRIX_FILE = 'input_classifier.txt'

# threshold for early stopping kmeans 
K_THRESH = 1

# name of the codebook file
CODEBOOK_FILE = 'codebook.txt'


# Hashing trick
def feature_hashing(features, size_f=100):
    h = FeatureHasher(n_features=size_f)
    f = h.transform(features)
    return f.toarray()

# Function to generate the code book of visual words
# ..
def gen_codebook():
    if len(sys.argv) < 3:
        print"Usage: ./feature_extraction dataset_path video_sequence_path"
        exit(1)

    print "Parsing params"
    datasetpath = sys.argv[1]

    nclusters = 0

    # getting the categories
    cats = get_classes(datasetpath)
    ncats = len(cats)
    print "searching for folders at " + datasetpath
    if ncats < 1:
        print"Wrong path! \n"
        exit(1)
         
    all_files = []
    all_files_labels = {}

    all_features = {}
    cat_label = {}

    for cat, label in zip(cats, range(ncats)):
        # path of the class
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

    # Obs,: all_features is a dict of form: 'image-path' : 'descriptor'
    print "computing the visual words via k-means"

    # passing to numpy array
    all_features_array = dict2numpy(all_features)
    
    # number of features
    nfeatures = all_features_array.shape[0]
    # number of clusters ( by convention, using the squared of the number of features)
    nclusters = int(sqrt(nfeatures))

    # Performs k-means on a set of observation vectors forming k clusters. This yields a
    # codebook mapping centroids to codes and vice versa. The k-means algorithm adjusts the
    # centroids until sufficient progress cannot be made, i.e. the change in distortion since
    # the last iteration is less than some threshold.
    #
    # obs : ndarray
    # Each row of the M by N array is an observation vector. The columns are the features
    # seen during each observation.
    #
    # iter : int
    # The number of times to run k-means, returning the codebook with the lowest distortion.
    # This argument is ignored if initial centroids are specified with an array for the
    # k_or_guess paramter. This parameter does not represent the number of iterations of the
    # k-means algorithm.

    # thresh : float
    # Terminates the k-means algorithm if the change in distortion since the last k-means
    # iteration is less than thresh.

    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)
    print "writing the codebook in file "
    f = open(CODEBOOK_FILE, 'wb')
    savetxt(f,codebook)

    return (codebook_exists, codebook, nclusters)


# Function to generate the histograms for video frames
# ..
def gen_histograms(nclusters,codebook,codebook_exists):
    
    print "Starting to generate histograms to video.."
    
    
    if not exists(sys.argv[2]):
        print "This path doesn't exist!"
        exit(1)
    
    # taking the video folder name
    video_name = sys.argv[2].split("/")
    video_name = video_name[-1].split(".")
    video_name = video_name[0]

    test_files = []
    test_features = {}
    test_frames_labels = {}

    testfolder_path = sys.argv[2]

    test_files = get_imgfiles(testfolder_path)
    # extracting features
    feats = extractSift(test_files)

    test_features.update(feats)

    # Computing the histograms based in codebook pre-processed (for all video frames )
    # histograms , dict of form: 'image_name' -> 'histogram'
    histograms = {}

    for imagefname in test_features:
        visual_histogram = computeHistograms(codebook, test_features[imagefname])
        histograms[imagefname] = visual_histogram

    print "writing histograms to file"
    path = "histograms"
    if not exists(path):
        mkdir(path)

    if not codebook_exists:
        number_of_words = nclusters
        writeHistogramsToFile(number_of_words,
                              test_files,
                              histograms,
                              path + "/" + video_name + '_' + HISTOGRAMS_FILE)
    else:
        content_file = open(CODEBOOK_FILE, 'r')
        number_of_words = len(content_file.readlines())
        writeHistogramsToFile(number_of_words,
                              test_files,
                              histograms,
                              path + "/" + video_name + '_' + HISTOGRAMS_FILE)

def hashing_trick():
    if isfile(sys.argv[2] + "_" + HISTOGRAMS_FILE):
        # apply feature hashing
        f = open(sys.argv[2] + "_" + HISTOGRAMS_FILE,'r')
        dict_list = []
        for _ in f:
            splited_t = _.split(" ")
            dict_words = {}
            index = 1
            for p in splited_t:
                dict_words[str(index)] = float(p)
                index+=1
            dict_list.append(dict_words)

        return feature_hashing(dict_list)
    return None

if __name__ == '__main__':
    codebook_exists = True
    nclusters = 0

    args = sys.argv

    # checking if already exist a codebook. If not, create a new codebook.
    # ...
    if not isfile(CODEBOOK_FILE):
        codebook_exists = False
        (codebook_exists, codebook, nclusters) = gen_codebook()

    else:

        print "There's a codebook. Using him"
        content_file = open(CODEBOOK_FILE, 'r')
        codebook = stringToNumpy(content_file)

        for line in content_file:
            nclusters = nclusters + 1
    
    # generating the histograms
    gen_histograms(nclusters, codebook, codebook_exists)


    # (codebook_exists, codebook, nclusters) = gen_codebook()

    # gen_histograms(nclusters, codebook, codebook_exists)

    # This matrix will be used by the classifier
    # hashMatrix = hashing_trick()

    # saving into a file
    # writeHashMatrixToFile(HASHMATRIX_FILE,hashMatrix)

    # .. plotting the histograms

    #for imageName in test_features:
    #    plt.hist(histograms[imageName])
    #    plt.title("Gaussian Histogram")
    #    plt.xlabel("Value")
    #    plt.ylabel("Frequency")
    #    plt.show()

