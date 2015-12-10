"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                         Video Processing Module                          |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2015-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: process_video.py
//  ========
//  Source file for process the group of frames that represent a video. The task
//  is trasform a video into a sequence of frames and save them in a folder appro-
//  priate.


// Parameters:
// sys.args[1] => class of the video
// sys.args[2] => name of the video

"""

def doc():
 	print (__doc__)

import numpy as np
import cv2
import sys
from os import mkdir
from os.path import splitext, exists

if len(sys.argv) < 3 or len(sys.argv) < 2:
	print"Usage: ./process_video video_class video_path"
	exit(1)
# getting the class of the video
clss = sys.argv[1]
# getting the name of video
vname = sys.argv[2]

# Opening the video
cap = cv2.VideoCapture(vname)

# Frame List
frames_ = []

# Reading the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
    	break

    frames_.append(frame)

    # Show the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

frame_count = 0
if not exists(clss):
    mkdir(clss)

filename = splitext(vname)[0]
    
for i in frames_:
	cv2.imwrite(clss + '/' + filename + str(frame_count) + '.png',i)
	frame_count+=1

cap.release()
cv2.destroyAllWindows()
