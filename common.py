"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                              Common Module                               |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2015-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
OVERVIEW: feature_extraction.py
//  ================================
//  This module implement methods that are common to another modules, by exam-
//  the feature_extraction.py.
//
"""


from os.path import exists, isdir, basename, isfile, join, splitext
from sklearn.feature_extraction import FeatureHasher
import sift
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, fromstring, asarray
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
from cPickle import dump, HIGHEST_PROTOCOL
import argparse
import sys
import cv2
import os
import time
from os import mkdir
from os.path import splitext, exists

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
PRE_ALLOCATION_BUFFER = 1000  # for sift


def generate_dataset(f):
    """lines = f.readlines()
    urls = []
    labels = []
    nmax = 10
    counter = 0 

    # Taking the urls and labels
    for line in lines:
        if counter == nmax:
            break
        urlnlab = asarray(line.split(" "))
        urls.append(urlnlab[0])
        labels.append(urlnlab[1])
        counter=counter+1
    
    counter=0
    if not exists("videos/"):
        mkdir("videos/")

    # Dowloading the videos by url ( Ps.: the videos aare saved with its line position on the file)
    for url in urls:
        cmnd = str("youtube-dl --output \"videos/" + str(counter) + ".mp4\" " + url)
        os.system(cmnd)
        counter=counter+1
 

    all_videos = []

    path = "videos"
    if not exists(path):
        mkdir(path)

    for fname in glob(path + "/*"):
        all_videos.extend([join(path, basename(fname))])
    
    path = "frames"
    if not exists(path):
        mkdir(path)

    for vid in (all_videos):
        vname = vid

        # Opening the video
        cap = cv2.VideoCapture(vname)

        # Frame List
        frames_ = []

        # Reading the video
        start = time.time()
        while(cap.isOpened()):
            ret, frame = cap.read()

            interval = time.time()
            if ret == False:
                break

            # 0.5 second window to take a frame
            if (interval - start) > 0.5:
                frames_.append(frame)
                start = interval

            # Show the frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count = 0

        filename = vname.split("/")
        filename = filename[1].split(".")
        filename = filename[0]

        if not exists(path + '/' + filename):
            mkdir(path + '/' + filename)
            
        for i in frames_:
            cv2.imwrite(path + '/' + filename + '/' + filename + '_' + str(frame_count) + '.png',i)
            frame_count+=1

        cap.release()
        cv2.destroyAllWindows()
    """
    path = "frames"
    all_videos = []
    if not exists(path):
        mkdir(path)
    for fname in glob(path + "/*"):
        all_videos.extend([join(path, basename(fname))])
    

    # Now, extracting the features for each video
    for vid in all_videos:
        filename = vid.split("/")
        filename = filename[1].split(".")
        filename = filename[0]

        cmnd = "python feature_extraction.py dataset " + path + '/' + filename
        os.system(cmnd)

# extracting the class names given a folder name (dataset)
def get_classes(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]

    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]

    return cats

# getting the array of files(images) inside a given folder
def get_imgfiles(path):
    all_files = []

    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

# calculate the sift descriptor for each image of a input array. The output
# is saved with the same name of image file plus '.sift'
def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}

    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'

        if exists(features_fname) == False:
            #print "calculating sift features for", fname
            sift.process_image(fname, features_fname)

        #print "gathering sift features for", fname,
        locs, descriptors = sift.read_features_from_file(features_fname)
        print descriptors.shape
        all_features_dict[fname] = descriptors

    return all_features_dict

# transforming a dict in a numpy array
# ...
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

# calculating histograms given a codebook that represents the vocabulary and
# the array of descriptors, generated by each image
def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

# writing the histograms into the file 
def writeHistogramsToFile(nwords, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label

    for fname in fnames:
        histogram = all_word_histgrams[fname]

        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
        
        data_row = hstack((0, histogram))
        data_rows = vstack((data_rows, data_row))
    
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    
    savetxt(features_fname, data_rows)


def writeHashMatrixToFile(filename, hashMatrix):
    savetxt(filename, hashMatrix)

# passing the codebook of string to numpy array
def stringToNumpy(codebook_file):
    codebook = []

    lines = codebook_file.readlines()

    for line in lines:
        line_array = fromstring(line,dtype=float,sep=' ')
        codebook.append(line_array)

    return asarray(codebook)