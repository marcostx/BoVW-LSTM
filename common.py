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
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, fromstring, asarray, array
import numpy as np
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

# dict for UCF classes
targets_ucf = {}

def download_videos(f):
    """

    Donwload each video according the urls into the file

    Each video are saved with the number of line in file f,
    easing to search for its class.
    
    """
    lines = f.readlines()
    urls = []
    labels = []
    nmax = 10
    # setting the maximum number of videos that will be downloaded
    max_videos = 100

    # Taking the urls and labels
    for line in lines:
        if counter == nmax:
            break
        urlnlab = asarray(line.split(" "))
        urls.append(urlnlab[0])
        labels.append(urlnlab[1])
        counter=counter+1
    
    counter=0
    path = "videos"
    if not exists(path):
        mkdir(path)

    # Dowloading the videos by url ( Ps.: the videos aare saved with its line position on the file)
    for url in urls:
        cmnd = str("youtube-dl --output \"videos/" + str(counter) + ".mp4\" " + url)
        os.system(cmnd)
        counter=counter+1

    all_videos = []

    # Getting each video name
    for fname in glob(path + "/*"):
        all_videos.extend([join(path, basename(fname))])
    
    path = "frames"
    if not exists(path):
        mkdir(path)

    # Computing the frames sequence for each video.
    # Saving into frames/ folder

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

def process_ucf_dataset(datasetpath):

    all_classes = []
    all_videos = []

    maxvideos_infolder = 5

    # Getting each class name
    for cl in glob(datasetpath + "/*"):
        all_classes.extend([join(datasetpath, basename(cl))])

    count_vid = 0
    # Getting each video name
    for i in range(len(all_classes)):
        for vid in glob(all_classes[i] + "/*"):

            if count_vid >= maxvideos_infolder:
                break

            all_videos.extend([join(all_classes[i], basename(vid))]) 
            count_vid += 1
        count_vid=0

    # Getting the frames for each video, using the script 'process_video.py'
    for vid in all_videos:
        cl = vid.split("/")[-2]
        # Taking the frames ..
        cmnd = "python process_video.py " + cl + ' ' + vid
        os.system(cmnd)
        

def generate_ucf_dataset(datasetpath):
    histograms_path = "histograms"
    all_classes = []
    # Getting each class name
    for cl in glob(path + "/*"):
        all_classes.extend([join(datasetpath, basename(cl))])


    # Extract the histograms (if necessary ..)
    if not exists(histograms_path):
        path = "frames"


        all_videos =  []
        
        # Getting each video name
        for i in range(len(all_classes)):
            for vid in glob(all_classes[i] + "/*"):
                all_videos.extend([join(all_classes[i], basename(vid))]) 

        # Now, extracting the features for each video
        for vid in all_videos:

            cmnd = "python feature_extraction.py dataset " + vid
            os.system(cmnd)

    X = []
    Y = []

    # fitting target vector
    for i in range(len(all_classes)):
        targets_ucf[basename(all_classes[i])] = i

    trainset_path = "histograms"

    # getting the histograms names
    hist_files = []
    for tname in glob(trainset_path + "/*"):
        hist_files.extend([join(trainset_path, basename(tname))])

    # Now we'll open each histogram and fill the X and Y vector. The line will be
    # the X value and the Y the value in the name position at targets_ucf vector.
    for hist_f in hist_files:
        histofile = open(hist_f)
        lines = histofile.readlines()

        label = targets_ucf[basename(hist_f.split("_")[1])]

        for i in range(len(lines)):
            X.append(lines[i])
            Y.append(label)

        histofile.close()
    
    # Passing the vector X to numpy array    
    for i in range(len(X)):
        splited = X[i].split(" ")
        splited_float = []
        for index in range(len(splited)-1):
            splited_float.append(splited[index])
        X[i] = np.asarray(splited_float,dtype='float')

    return X, Y



def generate_dataset(filename):
    f = open(filename)
    histograms_path = "histograms"

    # Extract the histograms (if necessary ..)
    if not exists(histograms_path):
        path = "frames"
        all_videos = []

        for fname in glob(path + "/*"):
            all_videos.extend([join(path, basename(fname))])
        

        # Now, extracting the features for each video
        for vid in all_videos:
            filename = vid.split("/")
            filename = filename[1].split(".")
            filename = filename[0]

            cmnd = "python feature_extraction.py dataset " + path + '/' + filename
            os.system(cmnd)

    # Creating the dataset (X and Y)
    X = []
    Y = []

    trainset_path = "histograms"
    trainset_files = []
    for tname in glob(trainset_path + "/*"):
        trainset_files.extend([join(trainset_path, basename(tname))])

    dataset_lines = f.readlines()
    # Filling the X and Y vectors
    for histofilename in trainset_files:

        # Searching the label in dataset sports-1m file
        path_splited = histofilename.split("/")
        path_splited = path_splited[1].split(".")
        line = path_splited[0].split("_")[0]

        string_line_dataset = dataset_lines[int(line)]
        label = string_line_dataset.split(" ")[1]

        # opening the histogram file of the video
        histofile = open(histofilename)
        lines = histofile.readlines()

        for i in range(len(lines)):
            #lines[i] = asarray(lines[i])
            X.append(lines[i])
            Y.append(label)

    histofile.close()

    for i in range(len(X)):
        splited = X[i].split(" ")
        splited_float = []
        for index in range(len(splited)-1):
            splited_float.append(splited[index])
        X[i] = np.array(splited_float,dtype='float')

    X = asarray(X)
    Y = map(int,Y)

    f.close()

    return X, Y
            

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
        
        # check if there is description for the image
        if len(descriptors) > 0:
            print descriptors.shape
            all_features_dict[fname] = descriptors

    return all_features_dict

# Transforming a dict in a numpy array
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