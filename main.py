import pandas as pd
from Utils import PreProcessData, TestTrainSplit
from Models import NaiveBayes, LogisticRegression, KNearestNeighbours, DecisionTree

pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 1000)


# ---------------------------Train Data set--------------------------------------------
# getting the train dataset
train_data = pd.read_csv('./Data/train.csv')
# preprocessing the Data
train_data['keyword'].fillna('', inplace=True)
train_data['location'].fillna('', inplace=True)
train_data['text_1'] = train_data['keyword'] + " " + train_data['location'] + " " + train_data['text']
train_data['text_1'] = train_data['text_1'].apply(PreProcessData.preProcessData)

# Splitting the Dataset at 80% to train and 20% to test
X = train_data['text_1']
y = train_data['target']
X_train, X_test, y_train, y_test = TestTrainSplit.getTestTrainSplit(X, y)

# -------------------------------------------------------------------------------------

# ------------------------------ Test Data set-----------------------------------------
# getting the test dataset
test_data = pd.read_csv('./Data/test.csv')

# preprocessing the Data
test_data['keyword'].fillna('', inplace=True)
test_data['location'].fillna('', inplace=True)
test_data['text_1'] = test_data['keyword'] + " " + test_data['location'] + " " + test_data['text']
test_data['text_1'] = test_data['text_1'].apply(PreProcessData.preProcessData)

# -------------------------------------------------------------------------------------

# ----------------------------------- Naive Bayes -------------------------------------

print('Naive Bayes Algorithm')
NaiveBayes.Algo(X_train, X_test, y_train, y_test, test_data)

# -------------------------------------------------------------------------------------

# ----------------------------------- Logistic Regression------------------------------

print('Logistic Regression Algorithm')
LogisticRegression.Algo(X_train, X_test, y_train, y_test, test_data)

# -------------------------------------------------------------------------------------

# ----------------------------------K Nearest Neighbours-------------------------------

print('K Nearest Neighbours Algorithm')
KNearestNeighbours.Algo(X_train, X_test, y_train, y_test, test_data)

# -------------------------------------------------------------------------------------

# -------------------------------------Decision Tree-----------------------------------

print('Decision Tree Algorithm')
DecisionTree.Algo(X_train, X_test, y_train, y_test, test_data)

# -------------------------------------------------------------------------------------
