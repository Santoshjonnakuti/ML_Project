def getInfo(dataFrame):
    print(dataFrame.info())


def getCountOfMissingValues(dataFrame):
    print(dataFrame.isna().sum())
