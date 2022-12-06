## Facial Recognition
Face recognition via deep learning hinges on a technique called deep metric learning.  In deep metric learning we output a real-valued feature vector.  Deep Metric learning combines metric and deep learning techniques.

Metric learning is an approach based directly on a distance metric that aims to establish similarity or dissimilarity between images (color detection and centroid tracking are examples of metric learning).

Deep Metric Learning uses Neural Networks to automatically learn discriminative features from the images and then compute the metric.

Deep Metric Learning helps capture Non-Linear feature structure by learning a non-linear transformation of the feature space.  Metric Learning is limited to just linear feature structures.

This repository leverages face_detection package from Adam Geitgey to do the facial detection, for more information on the project see go to this git repository https://github.com/ageitgey/face_recognition

## Facial Recognition Workflow
 Modern Face Recognition using Deep Metric Learning follows a four step deep learning process:

1. Face Detection (HOG or CNN)

2. Face Landmark Estimation - machine learning algorithm able to find 68 specific landmarks on a face

3. Encoding Faces - run the face images through a pre-trained network(DML) to get the 128 measurements for each face. The training process works by looking at 3 face images at a time:

    a. Load a training face image of a known person

    b. Load another picture of the same known person

    c. Load a picture of a totally different person

Then the algorithm looks at the measurements it is currently generating for each of those three images. It then tweaks the neural network slightly so that it makes sure the measurements it generates for image 1 and image 2 (same person) are slightly closer while making sure the measurements for image 2 and image 3 (different people) are slightly further apart

4. Take the measurements from a new image you want to identify and compare to measurements of  known persons and find the label of the closest match to identify the person
