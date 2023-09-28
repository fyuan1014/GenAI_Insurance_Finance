import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


import warnings
warnings.filterwarnings('ignore')

### 1. Function to detect columns with limited categories
def checkCountDistinct(df,catVars,threshold):
    catList=list()
    for cat in catVars:
        disCount=len(df[cat].value_counts())
        
        if disCount<threshold:
            catList.append(cat)
        else:
            pass
    
    return catList

### 2. Reading original data
claimFrequencyData=pd.read_csv(".../freMTPL2freq.csv")
creditData=pd.read_csv(".../creditcard.csv")

### 3. Spliting data into train and test
##### DrivAge is a sensitive variable, we drop it in training the synthesizer 
##### Here saved training and test data, where training data will be used for training the synthesizer and test will be used for evaluating synthetic data quality in the ctGANdataEvaluaiton_2.py
clmX = claimFrequencyData.drop(['ClaimNb','DrivAge'],axis=1)
clmY = claimFrequencyData['ClaimNb']
clmXTrain, clmXTest, clmYTrain, clmYTest = train_test_split(clmX, clmY, test_size=0.3, random_state=123)

claimFrequencyTrain = clmXTrain.copy()
claimFrequencyTrain['ClaimNb'] = clmYTrain
claimFrequencyTest = clmXTest.copy()
claimFrequencyTest['ClaimNb'] = clmYTest

claimFrequencyTrain.to_csv(".../freMTPL2freq_train.csv")
claimFrequencyTest.to_csv(".../freMTPL2freq_test.csv")

creditX = creditData.drop('Class',axis=1)
creditY = creditData['Class']
creditXTrain, creditXTest, creditYTrain, creditYTest = train_test_split(creditX, creditY, test_size=0.3, random_state=123)

creditDataTrain = creditXTrain.copy()
creditDataTrain['Class'] = creditYTrain
creditDataTest = creditXTest.copy()
creditDataTest['Class'] = creditYTest

creditDataTrain.to_csv(".../creditcard_train.csv")
creditDataTest.to_csv(".../creditcard_test.csv")

### 4. Detecting metadata (column/varaible types)
metadataClm = SingleTableMetadata()
metadataCredit = SingleTableMetadata()

metadataClm.detect_from_dataframe(data=claimFrequencyTrain)
metadataCredit.detect_from_dataframe(data=creditDataTrain)

#### 4.1 Revising columns types
feClmLst=claimFrequencyTrain.columns.to_list()
numToCatClm=checkCountDistinct(claimFrequencyTrain,feClmLst,threshold=15)

for cat in numToCatClm:
    metadataClm.update_column(column_name=cat,sdtype='categorical')

feCreditLst=creditDataTrain.columns.to_list()
numToCatCredit=checkCountDistinct(creditDataTrain,feCreditLst,threshold=15)

for cat in numToCatCredit:
    metadataCredit.update_column(column_name=cat,sdtype='categorical')

#### 4.2 Defining id column with policyID IDpol and setting it as primary key in the claimFrequency data
metadataClm.update_column(column_name='IDpol',sdtype='id')

metadataClm.set_primary_key(column_name='IDpol')

### 5. Creating and saving a CTGAN synthesizer

synthesizerClm = CTGANSynthesizer(metadataClm)
synthesizerClm.fit(claimFrequencyTrain)

synthesizerClm.save('.../ctganClmModel.pkl')

synthesizerCredit = CTGANSynthesizer(metadataCredit)
synthesizerCredit.fit(creditDataTrain)

synthesizer.save('.../ctganCreditModel.pkl')
