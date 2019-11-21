import cv2
import feature
import knn_classifier
import os
import os.path

source_image = cv2.imread('redshirt.jpg')
prediction = 'n.a.'
PATH = './training.data'
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    open('training.data', 'w')
    feature.training()
feature.test_image(source_image)
prediction = knn_classifier.main('training.data', 'test.data')
cv2.putText(
    source_image,
    'Prediction: ' + prediction,
    (35, 25),
    cv2.FONT_HERSHEY_PLAIN,
    1,
    50,
    )
cv2.imshow('color classifier', source_image)
cv2.waitKey(0)
