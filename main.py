import pandas as pd
from Utils import DataInsights, PreProcessData

# getting the train dataset
train_data = pd.read_csv('./Data/train.csv')
print('-'*30, 'Train Data', '-'*30)
print(train_data.head())
print('-'*30, 'Train Data Info', '-'*30)
print(DataInsights.getInfo(train_data))
print('-'*30, 'Missing Values', '-'*30)
# printing the train data info
print(DataInsights.getCountOfMissingValues(train_data))
# preprocessing the tweet text
train_data['text_1'] = train_data['text'].apply(PreProcessData.preProcessData)

print('-'*30, 'Train Data', '-'*30)
print(train_data.head())

# getting the test dataset
test_data = pd.read_csv('./Data/test.csv')
print('-'*30, 'Test Data', '-'*30)
print(test_data.head())
print('-'*30, 'Test Data Info', '-'*30)
print(DataInsights.getInfo(test_data))
print('-'*30, 'Missing Values', '-'*30)
# printing the test data info
print(DataInsights.getCountOfMissingValues(test_data))
# preprocessing the tweet text
test_data['text_1'] = test_data['text'].apply(PreProcessData.preProcessData)

print('-'*30, 'Test Data', '-'*30)
print(test_data.head())
