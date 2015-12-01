""" 
    
    Author: Jan Erik Solem, 2009-01-30
    Adapted by: Marcos Teixeira, 2015-12-01
    http://www.janeriksolem.net/2009/02/sift-python-implementation.html

"""

import os
import subprocess
from PIL import Image
from numpy import *
import numpy
import cPickle
import pylab
from os.path import exists
MAXSIZE = 1024
VERBOSE = True
WRITE_VERBOSE = False  # no verbose reading atm


def process_image(imagename, resultname='temp.sift', dense=False):    
    """ process an image and save the results in a .key ascii file"""
    print "working on ", imagename
    if dense == False:
        if imagename[-3:] != 'pgm':
            # create a pgm file, image is resized, if it is too big.
            # sift returns an error if more than 8000 features are found
            size = (MAXSIZE, MAXSIZE)
            im = Image.open(imagename).convert('L')
            im.thumbnail(size, Image.ANTIALIAS)
            im.save('tmp.pgm')
            imagename = 'tmp.pgm'
        
        cmmd = "./sift < " + imagename + " > " + resultname

        # run extraction command
        returnvalue = subprocess.call(cmmd, shell=True)
        if returnvalue == 127:
            os.remove(resultname) # removing empty resultfile created by output redirection
            raise IOError("SIFT executable not found")
        if returnvalue == 2:
            os.remove(resultname) # removing empty resultfile created by output redirection
            raise IOError("image " + imagename + " not found")            
        if os.path.getsize(resultname) == 0:
            raise IOError("extracting SIFT features failed " + resultname)

    else:
        import vlfeat

        # defines how dense the grid is
        size = (150, 150)
        step = 10
        
        im = Image.open(imagename).resize(size, Image.ANTIALIAS)
        im_array = numpy.asarray(im)
        if im_array.ndim == 3:
            im_gray = vlfeat.vl_rgb2gray(im_array)
        elif im_array.ndim == 2:
            im_gray = im_array
        else:
            raise IOError("Not enough dims found in image " + resultname)
        

        locs, int_descriptors = vlfeat.vl_dsift(im_gray, step=step, verbose=VERBOSE)
        nfeatures = int_descriptors.shape[1]
        padding = numpy.zeros((2, nfeatures))
        locs = numpy.vstack((locs, padding))
        header = ' '.join([str(nfeatures), str(128)])
        temp = int_descriptors.astype('float')  # convert descriptors to float
        descriptors = temp[:]
        with open(resultname, 'wb') as f:
            cPickle.dump([locs.T, descriptors.T], f, protocol=cPickle.HIGHEST_PROTOCOL)
        print "features saved in", resultname
        if WRITE_VERBOSE:
            with open(resultname, 'w') as f:
                f.write(header)
                f.write("\n")
                for i in range(nfeatures):
                    f.write(' '.join(map(str, locs[:, i])))
                    f.write("\n")
                    f.write(' '.join(map(str, descriptors[:, i])))
                    f.write("\n")


def read_features_from_file(filename='temp.sift', dense=False):
    """ read feature properties and return in matrix form"""
    
    if exists(filename) != False | os.path.getsize(filename) == 0:
        raise IOError("wrong file path or file empty: "+ filename)
    if dense == True:
        with open(filename, 'rb') as f:
            locs, descriptors = cPickle.load(f)
    else:           
        f = open(filename, 'r')
        header = f.readline().split()
        
        num = int(header[0])  # the number of features
        featlength = int(header[1])  # the length of the descriptor
        if featlength != 128:  # should be 128 in this case
            raise RuntimeError('Keypoint descriptor length invalid (should be 128).')
                 
        locs = zeros((num, 4))
        descriptors = zeros((num, featlength));        

        #parse the .key file
        e = f.read().split()  # split the rest into individual elements
        pos = 0
        for point in range(num):
            #row, col, scale, orientation of each feature
            for i in range(4):
                locs[point, i] = float(e[pos + i])
            pos += 4
            
            #the descriptor values of each feature
            for i in range(featlength):
                descriptors[point, i] = int(e[pos + i])
            #print descriptors[point]
            pos += 128
            
            #normalize each input vector to unit length
            descriptors[point] = descriptors[point] / linalg.norm(descriptors[point])
            #print descriptors[point]
            
        f.close()
    
    return locs, descriptors