import pandas as pd
from tqdm import tqdm
from pathlib import Path

csvFile = Path(__file__).parent.joinpath("data/poseLandmarkDataset/coordinate.csv")
df = pd.read_csv(csvFile)
columns = df.columns

carry = 1
targetDfIndex = [] # Index of where Y axis of a single landmark is
visibilityDict = {}
for index, column in enumerate(df.columns):
    if column.split("_")[-1] == "y":
        targetDfIndex.append(index)
        visibilityDict[index + carry] = []
        carry += 1
visibilityDictKeys = list(visibilityDict.keys())

# Calculate each row
for rowIndex in range(len(df)): # Row loop
    rowValues = df.iloc[rowIndex].values
    for index, targetIndex in enumerate(targetDfIndex): # Landmark loop
        # If both X and Y of a single landmark then the visibility of that landmark will be zero
        if rowValues[targetIndex] <= 0 and rowValues[targetIndex - 1] <= 0:
            #print(f"{rowIndex} {columns[targetIndex]} {columns[targetIndex - 1]}")
            visibilityDict[visibilityDictKeys[index]].append(0)
        else:
            visibilityDict[visibilityDictKeys[index]].append(1)

#print(visibilityDict)

# Insert the visibility to dataframe
for index, key in enumerate(visibilityDictKeys):
    columnName = columns[targetDfIndex[index]][:-1] + "visibility"
    df.insert(key, columnName, visibilityDict[key])

df.to_csv(csvFile, index=False)