#!usr/bin/python
# Imports
import csv
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

# Some global variables
t = Tokenizer()
CSVFile = "caucasian.csv"
CSVFileOutput = "test_output_7.csv"


# Functions
# Import CSV into a list, using the header name
def importcsvcol(CSVFileName, colName):
    csvFileName = open(CSVFileName, "r")
    csvFile = csv.DictReader(csvFileName)
    csvResList = []
    for col in csvFile:
        csvResList.append(col[colName])
    return csvResList


# Routine to add spaces to the characters of a word, so as to use the bag-of-words model
# As the tokeniser on texts-to-matrix likes to split on spaces
def addspaces(listtoaddspaces):
    newlist = []
    newoutput = []
    newlist = listtoaddspaces.copy()
    for item in newlist:
        newoutput.append(" ".join(item))
    return newoutput


# Take a list, and output to CSV and also print to screen for debugging
def PrintToScreenAndCSV(ListToOutput):
    CSVWrite.writerow(ListToOutput)
    for printitem in ListToOutput:
        print(printitem, end=" ")
    print()


# Main body of code

# Open CSV file for output
f = open(CSVFileOutput, "w", encoding="UTF8", newline="")
# Set up CSV
CSVWrite = csv.writer(f)

# Import two matching lines of text into two equal-length lists
# No error checking!
en = importcsvcol(CSVFile, "lang1")
de = importcsvcol(CSVFile, "lang2")
ne = importcsvcol(CSVFile, "Russian")
ze = importcsvcol(CSVFile, "Chechen")
ic = importcsvcol(CSVFile, "Kazakh")
go = importcsvcol(CSVFile, "Tsakhur")
hu = importcsvcol(CSVFile, "Hurrian")

cog = importcsvcol(CSVFile, "cog")


# Create input lists. Because of the error checking, doing all the pairs at the same time
# This could be made much better it not in a rush!

inputcombinedlist = []
outputcombinedlist = []
endelist = []
nedelist = []
zedelist = []
icdelist = []
godelist = []
hudelist = []

for itemone, itemtwo, itemthree in zip(en, de, cog):
    if itemthree != "x" or "":
        inputcombinedlist.append(itemone + "|" + itemtwo)
        endelist.append(itemone + "|" + itemtwo)
        outputcombinedlist.append(int(itemthree))

# This is horrible. Makes the test matrices of ne,ze etc paired with de
for cogvalue, itemone, itemtwo, itemthree, itemfour, itemfive, itemsix in zip(cog, de, ne, ze, ic, go, hu):
    if cogvalue != "x" or "":
        nedelist.append(itemtwo + "|" + itemone)
        zedelist.append(itemthree + "|" + itemone)
        icdelist.append(itemfour + "|" + itemone)
        godelist.append(itemfive + "|" + itemone)
        hudelist.append(itemsix + "|" + itemone)


# Split the data into four lists - 80% of each list into training, 20% into verification
splitpoint = int(len(inputcombinedlist) * 0.8)
InputTrain, InputValid = inputcombinedlist[:splitpoint], inputcombinedlist[splitpoint:]
OutputTrain, OutputValid = outputcombinedlist[:splitpoint], outputcombinedlist[splitpoint:]

# get all tokens from en and de into the tokeniser, so we can refer to each word as token
for item in inputcombinedlist:
    t.fit_on_texts(item)

for item in nedelist:
    t.fit_on_texts(item)

for item in zedelist:
    t.fit_on_texts(item)

for item in icdelist:
    t.fit_on_texts(item)

for item in godelist:
    t.fit_on_texts(item)

# Add spaces to deal with bag-of-words modelling
InputTrain = addspaces(InputTrain)
InputValid = addspaces(InputValid)

# Create the test matrices
EnTest = t.texts_to_matrix(addspaces(endelist), mode="count")
NeTest = t.texts_to_matrix(addspaces(nedelist), mode="count")
ZeTest = t.texts_to_matrix(addspaces(zedelist), mode="count")
IcTest = t.texts_to_matrix(addspaces(icdelist), mode="count")
GoTest = t.texts_to_matrix(addspaces(godelist), mode="count")
HuTest = t.texts_to_matrix(addspaces(hudelist), mode="count")


# Create Keras matrices, with appropriate training and validation data
kerasinputtrain = t.texts_to_matrix(InputTrain, mode="count")
kerasinputvalid = t.texts_to_matrix(InputValid, mode="count")


categorysize = kerasinputtrain.shape[1]
TrainSize = len(InputTrain)
ValidSize = len(InputValid)

# Preprocess the data (these are NumPy arrays)
kerasinputtrain = kerasinputtrain.reshape(TrainSize, categorysize).astype("float32")
kerasinputvalid = kerasinputvalid.reshape(ValidSize, categorysize).astype("float32")
kerasoutputtrain = np.array(OutputTrain, dtype=bool)
kerasoutputvalid = np.array(OutputValid, dtype=bool)


# categorysize= len(t.word_index)+1


# Create Model
# This could almost certainly be improved... a lot!
model = Sequential()
model.add(Dense(128, input_dim=categorysize, activation="gelu"))
model.add(Dense(128, activation="gelu"))
model.add(Dense(128, activation="gelu"))
model.add(Dense(1, activation="sigmoid"))


model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    loss=keras.losses.binary_crossentropy,
    metrics=keras.metrics.BinaryAccuracy(),
)

print(model.summary())


print("Fit model on training data")
history = model.fit(
    kerasinputtrain,
    kerasoutputtrain,
    batch_size=32,
    epochs=10000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(kerasinputvalid, kerasoutputvalid),
)


# Run the tests
# This is horrible, horrible code
# Can be massively cleaned up... at some point

PrintToScreenAndCSV(["Comparison", "Cognate? Model output", "Original En->De cognate answer"])

for checklist in [
    [EnTest, endelist, "langs1-2"],
    [NeTest, nedelist, "Russian"],
    [ZeTest, zedelist, "Chechen"],
    [IcTest, icdelist, "Kazakh"],
    [GoTest, godelist, "Tsakhur"],
    [HuTest, hudelist, "Hurrian"],
]:
    TestList = checklist[0]
    InputList = checklist[1]
    PrintToScreenAndCSV([checklist[2]])
    PrintToScreenAndCSV([""])
    for checkvalue in range(1, len(TestList)):
        # Prepare output variables
        modelresulttemp = model.predict(TestList[(checkvalue - 1) : checkvalue])
        modelresult = float(modelresulttemp[0])
        testitem = InputList[(checkvalue - 1) : checkvalue]
        resultitem = outputcombinedlist[(checkvalue - 1) : checkvalue]
        originalresult = bool(resultitem[0])

        PrintToScreenAndCSV([testitem[0], modelresult, originalresult])
        # print(InputList[(checkvalue-1):checkvalue])
        # print(modelresulttemp)
        # print("Correct answer for En-> DE should be",outputcombinedlist[(checkvalue-1):checkvalue])
        # print("==========================")
        # print()

    PrintToScreenAndCSV([""])
    PrintToScreenAndCSV([""])


# Close CSV writer file
f.close
exit()
