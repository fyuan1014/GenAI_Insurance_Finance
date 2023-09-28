import pandas as pd
import numpy as np
import os 

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from catboost import CatBoostClassifier

### 1. Functions
def checkCountDistinct(df,catVars,threshold):
    catList=list()
    for cat in catVars:
        disCount=len(df[cat].value_counts())
        
        if disCount<threshold:
            catList.append(cat)
        else:
            pass
    
    return catList

def performanceReport(yTrue,yPredict,labelsDefault = [1]):
    precision, recall, fscore, _ = precision_recall_fscore_support(yTrue, yPredict, average=None, labels = labelsDefault)
    return precision, recall, fscore

### 2. Loading data
claimFrequencyDataTrain=pd.read_csv(".../freMTPL2freq_train.csv")
creditDataTrain=pd.read_csv(".../creditcard_train.csv")

syntheticClmData=pd.read_csv(".../freMTPL2freq_train_synthetic.csv")
syntheticCreditData=pd.read_csv(".../creditcard_train_synthetic.csv")

claimFrequencyTest=pd.read_csv(".../freMTPL2freq_test.csv")
creditDataTest=pd.read_csv(".../creditcard_test.csv")

#### 2.1 Processing target variables of claim data
##### Here we make ClaimNb as a binary variable indicating whether a policy hae been involved with claims: if ClaimInd=1 (ClaimNb>0), this policy has been involved with claim; otherwise, it has not been involved with claim.

claimFrequencyDataTrain['ClaimInd']=np.where(claimFrequencyDataTrain['ClaimNb']>0,1,0)
syntheticClmData['ClaimInd']=np.where(syntheticClmData['ClaimNb']>0,1,0)
claimFrequencyTest['ClaimInd']=np.where(claimFrequencyTest['ClaimNb']>0,1,0)

clmFrequencyTrain=claimFrequencyDataTrain.drop('ClaimNb',axis=1)
syntheticClmFrequency=syntheticClmData.drop('ClaimNb',axis=1)
clmFrequencyTest=claimFrequencyTest.drop('ClaimNb',axis=1)

#### 2.2 Processing categorical variables
#### Identifying categorical columns
columnsNotUsedClm=['IDpol','ClaimInd']
feClmLst=[ele for ele in clmFrequencyTrain.columns.to_list() if ele not in columnsNotUsedClm]
numToCatClm=checkCountDistinct(clmFrequencyTrain,feClmLst,threshold=15)
##### 'Region' has more than 15 categories, while it is categorical
numToCatClm=numToCatClm+['Region']

columnsNotUsedCredit=['Class']
feCreditLst=[ele for ele in creditDataTrain.columns.to_list() if ele not in columnsNotUsedCredit]
numToCatCredit=checkCountDistinct(creditDataTrain,feCreditLst,threshold=15)


### 3. Spliting data
x_testClm = clmFrequencyTest[feClmLst]
y_testClm = clmFrequencyTest['ClaimInd']

x_testCredit = creditDataTest[feCreditLst]
y_testCredit = creditDataTest['Class']

x_trainRealClm, x_valRealClm, y_trainRealClm, y_valRealClm = train_test_split(clmFrequencyTrain[feClmLst],clmFrequencyTrain['ClaimInd'], test_size=0.3, random_state=123)
x_trainSynClm, x_valSynClm, y_trainSynClm, y_valSynClm = train_test_split(syntheticClmFrequency[feClmLst],syntheticClmFrequency['ClaimInd'], test_size=0.3, random_state=123)

x_trainRealCredit, x_valRealCredit, y_trainRealCredit, y_valRealCredit = train_test_split(creditDataTrain[feCreditLst],creditDataTrain['Class'], test_size=0.3, random_state=123)
x_trainSynCredit, x_valSynCredit, y_trainSynCredit, y_valSynCredit = train_test_split(syntheticCreditData[feCreditLst],syntheticCreditData['Class'], test_size=0.3, random_state=123)


### 4. Training model
clfRealClm = CatBoostClassifier()
clfSynClm = CatBoostClassifier()
clfRealClm.set_params(train_dir='/tmp/catboost_info')
clfSynClm.set_params(train_dir='/tmp/catboost_info')

clfRealCredit = CatBoostClassifier()
clfSynCredit = CatBoostClassifier()
clfRealCredit.set_params(train_dir='/tmp/catboost_info')
clfSynCredit.set_params(train_dir='/tmp/catboost_info')

#### 4.1 Fitting claim catoost model with real&Synthetic data
evalRealClm = [(x_valRealClm, y_valRealClm)]
evalSynClm = [(x_valSynClm, y_valSynClm)]

clfRealClm.fit(x_trainRealClm, y_trainRealClm, cat_features=numToCatClm, eval_set=evalRealClm, verbose=False)
clfSynClm.fit(x_trainSynClm, y_trainSynClm, cat_features=numToCatClm, eval_set=evalSynClm, verbose=False)

#### 4.2 Fitting credit card catoost model with real&synthetic data
evalRealCredit = [(x_valRealCredit, y_valRealCredit)]
evalSynCredit = [(x_valSynCredit, y_valSynCredit)]

clfRealCredit.fit(x_trainRealCredit, y_trainRealCredit, cat_features=numToCatCredit, eval_set=evalRealCredit, verbose=False)
clfSynCredit.fit(x_trainSynCredit, y_trainSynCredit, cat_features=numToCatCredit, eval_set=evalSynCredit, verbose=False)


### 5. Generating model performance metrics
#### Here used 80 percentaile of predicted probability in train data to generate the performance metric

yRealClmTrainPredictProbability = clfRealClm.predict_proba(x_trainRealClm)[0:,1]
thresholdRealClmTrain = np.percentile(yRealClmTrainPredictProbability,80)
yRealClmTestPredictProbability = clfRealClm.predict_proba(x_testClm)[0:,1]
yRealClmTestPredict = np.where(yRealClmTestPredictProbability>thresholdRealClmTrain,1,0)

ySynClmTrainPredictProbability = clfSynClm.predict_proba(x_trainSynClm)[0:,1]
thresholdSynClmTrain = np.percentile(ySynClmTrainPredictProbability,80)
ySynClmTestPredictProbability = clfRealClm.predict_proba(x_testClm)[0:,1]
ySynClmTestPredict = np.where(ySynClmTestPredictProbability>thresholdSynClmTrain,1,0)

print("Claim model (trained based on real data) performance on claim test data are: ")
print("Precision: %.4g" %performanceReport(y_testClm, yRealClmTestPredict, labelsDefault = [1])[0])
print("Recall:  %.4g" %performanceReport(y_testClm, yRealClmTestPredict, labelsDefault = [1])[1])
print("F1 Score:  %.4g" %performanceReport(y_testClm, yRealClmTestPredict, labelsDefault = [1])[2])

print("Claim model (trained based on synthetic data) performance on claim test data are: ")
print("Precision: %.4g" %performanceReport(y_testClm, ySynClmTestPredict, labelsDefault = [1])[0])
print("Recall:  %.4g" %performanceReport(y_testClm, ySynClmTestPredict, labelsDefault = [1])[1])
print("F1 Score:  %.4g" %performanceReport(y_testClm, ySynClmTestPredict, labelsDefault = [1])[2])


yRealCreditTrainPredictProbability = clfRealCredit.predict_proba(x_trainRealCredit)[0:,1]
thresholdRealCreditTrain = np.percentile(yRealCreditTrainPredictProbability,80)
yRealCreditTestPredictProbability = clfRealCredit.predict_proba(x_testCredit)[0:,1]
yRealCreditTestPredict = np.where(yRealCreditTestPredictProbability>thresholdRealCreditTrain,1,0)

ySynCreditTrainPredictProbability = clfSynCredit.predict_proba(x_trainSynCredit)[0:,1]
thresholdSynCreditTrain = np.percentile(ySynCreditTrainPredictProbability,80)
ySynCreditTestPredictProbability = clfSynCredit.predict_proba(x_testCredit)[0:,1]
ySynCreditTestPredict = np.where(ySynCreditTestPredictProbability>thresholdSynCreditTrain,1,0)

print("Credit card fraud model (trained based on real data) performance on credit card test data are: ")
print("Precision: %.4g" %performanceReport(y_testCredit, yRealCreditTestPredict, labelsDefault = [1])[0])
print("Recall:  %.4g" %performanceReport(y_testCredit, yRealCreditTestPredict, labelsDefault = [1])[1])
print("F1 Score:  %.4g" %performanceReport(y_testCredit, yRealCreditTestPredict, labelsDefault = [1])[2])

print("Credit card fraud model (trained based on synthetic data) performance on credit card test data are: ")
print("Precision: %.4g" %performanceReport(y_testCredit, ySynCreditTestPredict, labelsDefault = [1])[0])
print("Recall:  %.4g" %performanceReport(y_testCredit, ySynCreditTestPredict, labelsDefault = [1])[1])
print("F1 Score:  %.4g" %performanceReport(y_testCredit, ySynCreditTestPredict, labelsDefault = [1])[2])