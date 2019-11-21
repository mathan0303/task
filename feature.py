from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import knn_classifier as knn_classifier


def test_image(test_image):

    # load the image
    image = test_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            print(feature_data)

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def training_image(img_name):

    # detect image color by using image file name to label training data
    if 'royal_blue' in img_name:
        data_source = 'royal_blue'
    elif 'royal_red' in img_name:
        data_source = 'royal_red'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    # red color training images
    for f in os.listdir('./training_dataset/royal_blue'):
        training_image('./training_dataset/royal_blue/' + f)

    for f in os.listdir('./training_dataset/royal_red'):
        training_image('./training_dataset/royal_red/' + f)