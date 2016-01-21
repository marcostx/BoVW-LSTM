Bag of Visual Words Image Feature Generator
============================================

This is an implementation of [bag of visual words model][1] in Python for feature extraction in videos.

The current repository is just one layer of a framework for video classification, composed by:
- Bag-of-Visual-Words ( Feature extraction for each frame) 
- Long-Short Term Memory ( Maximizing Temporal Dependencies of features)
- Softmax Classifier ( Classify the video, given the outputs of LSTM)

The first part consists of given an input video, split it up into sequence of frames and save these images in a folder that 
represents the video class. After this we extract the features for each image into the folder of video, generating a Histogram
of Visual Words belonging to each image. The first part is done by [`process_video.py`][2] and the second by [`feature_extraction.py`][3]

The script [`feature_extraction.py`][3] will generate a visual vocabulary using the images provided by process_video.py.

The feature extraction consists of:

1. Extracting local features of all datasets
2. Generating a codebook of visual words with clustering of the features
3. Aggregating the histograms of the visual words for each of the training images

This code relies on:

 - SIFT features for local features (external implementation)
 - k-means for generation of the words via clustering

### Example: Processing Video
To extract the frame sequence of a given video input eg. video.mp4 that belongs to a given class eg. walking, use this command:
    python process_video.py walking video.mp4
    
Then will be create a new folder with the same name of the class given containing each frame extracted
 

### Example: Extrating Features
You can extract the features of a video passing the path for the folder containing the video frames (eg. walking) extracted before by [`process_video.py`][2] script and passing what will be the dataset folder (eg. dataset_folder) used to generate (or simply use) the codebook. Use this command: 

    python feature_extraction.py dataset_folder/ walking/

The dataset should have following structure, where all the video frames belonging to one class are in the same folder:

    .
    |-- path_to_folders_with_video_frames
    |    |-- class1
    |    |-- class2
    |    |-- class3
    ...
    |    └-- classN

### Prerequisites:

To install the necessary libraries run following code from working directory:
    
    # installing sift
    wget http://www.cs.ubc.ca/~lowe/keypoints/siftDemoV4.zip
    unzip siftDemoV4.zip
    cp sift*/sift sift
    

#### Notes
If you get an `IOError: SIFT executable not found` error, try `sudo apt-get install libc6-i386`.
    
### References:

#### SIFT:
David G. Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision, 60, 2 (2004), pp. 91-110.

#### sift.py:
Taken from http://www.janeriksolem.net/2009/02/sift-python-implementation.html

[1]: https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision
[2]: https://github.com/marcostx/bag-of-visual-words/blob/master/process_video.py
[3]: https://github.com/marcostx/bag-of-visual-words/blob/master/feature_extraction.py
