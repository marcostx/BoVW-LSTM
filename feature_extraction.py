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
//  ========


// Parameters:
// sys.argv[1] => dataset path
// sys.argv[2] => test folder path
// sys.argv[3] => video name

"""

from os.path import exists, isdir, basename, join, splitext
import sift
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from cPickle import dump, HIGHEST_PROTOCOL
import argparse
import sys

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = '../dataset'
PRE_ALLOCATION_BUFFER = 1000  # for sift
HISTOGRAMS_FILE = 'trainingdata.lstm'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'

def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]

    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]

    return cats

def get_imgfiles(path):
    all_files = []

    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}

    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'

        if exists(features_fname) == False:
            print "calculating sift features for", fname
            sift.process_image(fname, features_fname)

        print "gathering sift features for", fname,
        locs, descriptors = sift.read_features_from_file(features_fname)
        print descriptors.shape
        all_features_dict[fname] = descriptors

    return all_features_dict

def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    
    savetxt(features_fname, data_rows, fmt)

if __name__ == '__main__':

    print "Parsing params"
    datasetpath = sys.argv[1]

    cats = get_categories(datasetpath)
    ncats = len(cats)
    print "searching for folders at " + datasetpath
    if ncats < 1:
        print"Wrong path! \n"
        exit(1)

    # using a train dataset to generate the codebook

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
    nclusters = int(sqrt(nfeatures))
    
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    print "writing the codebook in file "
    f = open(datasetpath + CODEBOOK_FILE, 'wb')
    dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    #print "compute the visual words histograms for each image"
    #all_word_histgrams = {}
    #for imagefname in all_features:
    #    word_histgram = computeHistograms(codebook, all_features[imagefname])
    #    all_word_histgrams[imagefname] = word_histgram

    #print "writing histograms to file"
    #writeHistogramsToFile(nclusters,
    #                      all_files_labels,
    #                      all_files,
    #                      all_word_histgrams,
    #                      datasetpath + HISTOGRAMS_FILE)

    # test in a new sequence of video frames
    print "Starting to generate histograms to video.."
    # taking the video folder name
    
    if not exists(sys.argv[2] + '/' + sys.argv[3]):
    	print "This path doesn't exist!"
    	exit(1)
    
    videofolder = sys.argv[3]
    
    test_files = []
    test_features = {}
    test_frames_labels = {}

    testfolder_path = join(sys.argv[2], videofolder)

    test_files = get_imgfiles(testfolder_path)
    # extracting features
    feats = extractSift(test_files)

    test_features.update(feats)
    
    # default class for all frames ( 0 )
    for i in cat_files:   
        test_frames_labels[i] = 0

    # Generate the histograms based on codebook pre-processed (for all frames of the video )
    histograms = {}
    for imagefname in test_features:
        visual_histogram = computeHistograms(codebook, test_features[imagefname])
        histograms[imagefname] = visual_histogram

#    print "writing histograms to file"
#    writeHistogramsToFile(nclusters,
#                          test_frames_labels,
#                          test_files,
#                          histograms,
#                          sys.argv[2] + HISTOGRAMS_FILE)

