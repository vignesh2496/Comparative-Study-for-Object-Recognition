import cv2
import os
import numpy as np
from scipy.spatial import distance

def get_images(categories):
    images = []
    for category in categories:
        folder_name = '../101_ObjectCategories/' + category
        images.append(len(os.listdir(folder_name)))
        for file_name in os.listdir(folder_name):
            img = cv2.imread(os.path.join(folder_name, file_name))
            images.append(img)
    return images

def split_images(images, train_split):
    train_images = []
    test_images = []
    Y_train = []
    Y_test = []
    category = 0
    i = 0
    while(i < len(images)):
        for j in range(i + 1, i + 1 + int(train_split * images[i])):
            train_images.append(images[j])
            Y_train.append(category)
        for j in range(i + 1 + int(train_split * images[i]), i + 1 + images[i]):
            test_images.append(images[j])
            Y_test.append(category)
        category = category + 1
        i = i + 1 + images[i]
    return np.array(train_images), np.array(Y_train), np.array(test_images), np.array(Y_test)

def extract_SIFT_features(images):
    SIFT_features_many_images = []
    for image in images:
        sift = cv2.xfeatures2d.SIFT_create()
        key_points, description = sift.detectAndCompute(image, None)
        SIFT_features_many_images.append(description)
    return SIFT_features_many_images

def convert_to_bag_of_words(SIFT_many_images, centroids):
    centroids = list(centroids)
    num_centroids = len(centroids)
    bag_of_words = []
    for SIFT_single_image in SIFT_many_images: 
        num_points = SIFT_single_image.shape[0]
        histogram = np.zeros(num_centroids)
        for point in SIFT_single_image:
            distances = [distance.euclidean(point, centroid) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            histogram[cluster_idx] = histogram[cluster_idx] + 1
        bag_of_words.append(histogram / num_points)
    return np.array(bag_of_words)

def train_test_split(images, train_split, K):
    train_images, Y_train, test_images, Y_test = split_images(images, train_split)
    SIFT_train = extract_SIFT_features(train_images)
    SIFT_all_images = []
    for SIFT_single_image in SIFT_train:
        for point in SIFT_single_image:
            SIFT_all_images.append(point)
    SIFT_all_images = np.array(SIFT_all_images)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centroids = cv2.kmeans(SIFT_all_images, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    SIFT_test = extract_SIFT_features(test_images)
    X_train = convert_to_bag_of_words(SIFT_train, centroids)
    X_test = convert_to_bag_of_words(SIFT_test, centroids)
    return X_train, Y_train, X_test, Y_test

images = get_images(['Faces_easy', 'airplanes', 'Motorbikes'])
X_train, Y_train, X_test, Y_test = train_test_split(images, train_split = 0.5, K = 70)
np.save('../Extracted_Features/X_train.npy', X_train)
np.save('../Extracted_Features/Y_train.npy', Y_train)
np.save('../Extracted_Features/X_test.npy', X_test)
np.save('../Extracted_Features/Y_test.npy', Y_test)
