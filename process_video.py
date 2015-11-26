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

"""

print (__doc__)

import numpy as np
import cv2

# Opening the video
cap = cv2.VideoCapture('test.mp4')

# Frame List
frames_ = []

# Reading the video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
    	break

    frames_.append(frame)

    # Show the frame
    # cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i in frames_:
	cv2.imshow('frame',i)

	if cv2.waitKey(20) == ord('q'):
		exit(1)

cap.release()
cv2.destroyAllWindows()