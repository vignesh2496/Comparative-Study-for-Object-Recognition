import numpy as np
import _pickle
import matplotlib.pyplot as plt
import extras 
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision = 2)

# Load train data
X_test = np.load('../Extracted_Features/X_test.npy')
Y_test = np.load('../Extracted_Features/Y_test.npy') 

# Test using Gaussian Naive Bayes Classifier
with open('../Trained_Classifiers/GNB_classifier.pkl', 'rb') as GNB_file:
    GNB_classifier = _pickle.load(GNB_file)
Y_pred_GNB = GNB_classifier.predict(X_test)
GNB_acc = np.sum(Y_pred_GNB == Y_test) / Y_test.shape[0] * 100
print("GAUSSIAN NAIVE BAYES CLASSIFIER")
print("-------------------------------")
print("GNB Accuracy = %.2f %%" % GNB_acc)
GNB_confusion_matrix = confusion_matrix(Y_test, Y_pred_GNB)
# Plot Normalized Confusion Matrix
plt.figure()
extras.plot_confusion_matrix(GNB_confusion_matrix, classes = ['Face', 'Airplane', 'Motorcycle'], normalize = True, title = 'Normalized confusion matrix')
plt.savefig('../Results/Varying_train_size/GNB_confusion_matrix_train_size_07.png')

# Test using Decision Tree Classifier
with open('../Trained_Classifiers/DTree_classifier.pkl', 'rb') as DTree_file:
    DTree_classifier = _pickle.load(DTree_file)
Y_pred_DTree = DTree_classifier.predict(X_test)
DTree_acc = np.sum(Y_pred_DTree == Y_test) / Y_test.shape[0] * 100
print("\n\nDECISION TREE CLASSIFIER")
print("------------------------")
print("DTree Accuracy = %.2f %%" % DTree_acc)
DTree_confusion_matrix = confusion_matrix(Y_test, Y_pred_DTree)
# Plot Normalized Confusion Matrix
plt.figure()
extras.plot_confusion_matrix(DTree_confusion_matrix, classes = ['Face', 'Airplane', 'Motorcycle'], normalize = True, title = 'Normalized confusion matrix')
plt.savefig('../Results/Varying_train_size/DTree_confusion_matrix_train_size_07.png')
