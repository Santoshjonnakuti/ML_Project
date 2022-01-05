from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from Utils import Metrics


def trainModel(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


def predict(model, X_test):
    y_predicted = model.predict(X_test)
    return y_predicted


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


def Algo(X_train, X_test, y_train, y_test, test_data):
    naiveBayesModel = trainModel(X_train, y_train)
    print('Model Trained...')
    y_predicted = predict(naiveBayesModel, X_test)
    print('Calculating Accuracy and CM....')
    # printing the metrics
    Metrics.findAccuracy(y_predicted, y_test)
    Metrics.getConfusionMatrix(y_predicted, y_test)
    # testing on new data
    test_y_predicted = predictForNewData(naiveBayesModel, test_data)
    print(test_data.head(50))
