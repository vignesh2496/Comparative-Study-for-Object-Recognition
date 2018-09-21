import numpy as np
import _pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# Load train data
X_train = np.load('../Extracted_Features/X_train.npy')
Y_train = np.load('../Extracted_Features/Y_train.npy') 

# Train Gaussian Naive Bayes Classifier  
GNB_classifier = GaussianNB()
GNB_classifier.fit(X_train, Y_train)
# Save the classifier
with open('../Trained_Classifiers/GNB_classifier.pkl', 'wb') as GNB_file:
    _pickle.dump(GNB_classifier, GNB_file) 

# Train Decision Tree Classifier
DTree_classifier = tree.DecisionTreeClassifier()
DTree_classifier.fit(X_train, Y_train)
# Save the classifier
with open('../Trained_Classifiers/DTree_classifier.pkl', 'wb') as DTree_file:
    _pickle.dump(DTree_classifier, DTree_file) 
