import pandas as pd
from Utils import DataInsights

# getting the train dataset
train_data = pd.read_csv('./Data/train.csv')
print('-'*30, 'Train Data', '-'*30)
print(train_data.head())
print('-'*30, 'Train Data Info', '-'*30)
print(DataInsights.getInfo(train_data))

# getting the test dataset
test_data = pd.read_csv('./Data/test.csv')
print('-'*30, 'Test Data', '-'*30)
print(test_data.head())
print('-'*30, 'Test Data Info', '-'*30)
print(DataInsights.getInfo(train_data))
