import pandas as pd


def importTrainData():
    trainData = pd.read_csv("/../Data/train.csv")
    return trainData


def importTestData():
    testData = pd.read_csv('/../Data/test.csv')
    return testData
