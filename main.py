from Utils import importData

train_data = importData.importTrainData()
print('This is Train Data')
print(train_data.head())

test_data = importData.importTestData()
print('This is Test Data')
print(test_data.head())
