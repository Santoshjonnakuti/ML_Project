from sklearn import metrics


# to find the accuracy of the model
def findAccuracy(y_test, predicted):
    accuracy = metrics.accuracy_score(y_test, predicted)
    print('Accuracy : ', str(round(accuracy*100)))
    return


# to find the confusion matrix of the model
def getConfusionMatrix(y_test, predicted):
    confusionMatrix = metrics.confusion_matrix(y_test, predicted)
    print('Confusion Matrix :')
    print(confusionMatrix)
    return
