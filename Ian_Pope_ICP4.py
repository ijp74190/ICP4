#Ian Pope 700717419
#ICP 4

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

#Data Processing and setup
glass_data = pd.read_csv('glass.csv')
#print(glass_data.head())
#print(glass_data.describe())
X = glass_data.iloc[:, :-1].values
y = glass_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Complement Naive Bayes
print("Complement Naive Bayes")

classifier = ComplementNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0.0),"\n")

print('accuracy is',accuracy_score(y_pred,y_test),"\n")

display = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()
plt.title('Complement Naive Bayes Confusion Matrix')
plt.show()

#Charts
#CNB Predicted Outputs
slices = [0,0,0,0,0,0,0]
for label in y_pred:
    slices[label-1] += 1
labels = ['1','2','3','4','5','6','7']
plt.pie(slices, labels=labels,
        wedgeprops={'edgecolor': 'black'})
plt.title('Complement Naive Bayes Predicted Outputs')
plt.tight_layout()
plt.show()

#Actual Outputs
slices = [0,0,0,0,0,0,0]
for label in y_test:
    slices[label-1] += 1
labels = ['1','2','3','4','5','6','7']
plt.pie(slices, labels=labels,
        wedgeprops={'edgecolor': 'black'})
plt.title('Complement Naive Bayes Actual Outputs')
plt.tight_layout()
plt.show()



#Linear SVC
print("\nLinear SVC:")
svc = LinearSVC(dual=False)
#svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print("svm accuracy of training data=", acc_svc)

print()
print(classification_report(y_test, y_pred, zero_division=0.0),"\n")
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test),"\n")


#Charts:
#Confusion Matrix SVC
display = ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()
plt.title('SVC Confusion Matrix')
plt.show()


#Pie Chart Predicted SVC
slices = [0,0,0,0,0,0,0]
for label in y_pred:
    slices[label-1] += 1
labels = ['1','2','3','4','5','6','7']
plt.pie(slices, labels=labels,
        wedgeprops={'edgecolor': 'black'})
plt.title('SVC Predicted Outputs')
plt.tight_layout()
plt.show()

#Pie Chart Actual
slices = [0,0,0,0,0,0,0]
for label in y_test:
    slices[label-1] += 1
labels = ['1','2','3','4','5','6','7']
plt.pie(slices, labels=labels,
        wedgeprops={'edgecolor': 'black'})
plt.title('SVC Actual Outputs')
plt.tight_layout()
plt.show()












