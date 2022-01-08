from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from Utils import Metrics


# to train the model
def trainModel(X_train, y_train):
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    return classifier


# to predict for the test data
def predict(model, X_test):
    y_predicted = model.predict(X_test)
    return y_predicted


# to predict for test.csv
def predictForNewData(model, test_data):
    corpus = []
    X = test_data['text_1']
    dataFrameLen = len(X)
    for i in range(dataFrameLen):
        corpus.append(X[i])
    cv = CountVectorizer(max_features=1700)
    X = cv.fit_transform(corpus).toarray()
    test_data['predicted'] = predict(model, X)
    return test_data['predicted']


# algorithm
def Algo(X_train, X_test, y_train, y_test, test_data):
    kNNModel = trainModel(X_train, y_train)
    y_predicted = predict(kNNModel, X_test)
    # printing the metrics
    Metrics.findAccuracy(y_test, y_predicted)
    Metrics.getConfusionMatrix(y_test, y_predicted)
    # testing on new data
    test_y_predicted = predictForNewData(kNNModel, test_data)
    print(test_data.head(30))