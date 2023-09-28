import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, get_column_plot, get_column_pair_plot
from sdv.metadata import SingleTableMetadata

import warnings
warnings.filterwarnings('ignore')

#### 1. Function to detect columns with limited categories
def checkCountDistinct(df,catVars,threshold):
    catList=list()
    for cat in catVars:
        disCount=len(df[cat].value_counts())
        
        if disCount<threshold:
            catList.append(cat)
        else:
            pass
    
    return catList


### 2. Loading real and synthetic data

claimFrequencyDataTrain=pd.read_csv(".../freMTPL2freq_train.csv")
creditDataTrain=pd.read_csv(".../creditcard_train.csv")

syntheticClmData=pd.read_csv(".../freMTPL2freq_train_synthetic.csv")
syntheticCreditData=pd.read_csv(".../creditcard_train_synthetic.csv")


### 3. Synthetic data evaluation
#### 3.1 Getting metadata of real data

metadataClm = SingleTableMetadata()
metadataCredit = SingleTableMetadata()

metadataClm.detect_from_dataframe(data=claimFrequencyDataTrain)
metadataCredit.detect_from_dataframe(data=creditDataTrain)

#### 3.2 Revising column types

feClmLst=claimFrequencyDataTrain.columns.to_list()
numToCatClm=checkCountDistinct(claimFrequencyDataTrain,feClmLst,threshold=15)

for cat in numToCatClm:
    metadataClm.update_column(column_name=cat,sdtype='categorical')

feCreditLst=creditDataTrain.columns.to_list()
numToCatCredit=checkCountDistinct(creditDataTrain,feCreditLst,threshold=15)

for cat in numToCatCredit:
    metadataCredit.update_column(column_name=cat,sdtype='categorical')

#### 3.3 Defining id column with policyID IDpol and setting it as primary key in the claimFrequency data
metadataClm.update_column(column_name='IDpol',sdtype='id')

metadataClm.set_primary_key(column_name='IDpol')


#### 3.4 Generating comparison report
qualityClmReport = evaluate_quality(
    claimFrequencyDataTrain,
    syntheticClmData,
    metadataClm
)

qualityCreditReport = evaluate_quality(
    creditDataTrain,
    syntheticCreditData,
    metadataCredit
)

#### 3.5 Visualizing comparisons
##### Here selected 'ClaimNb' from claim dataset, and 'Class' from credit card dataset as demonstration examples.
figClm = get_column_plot(
    real_data=claimFrequencyDataTrain,
    synthetic_data=syntheticClmData,
    column_name='ClaimNb',
    metadata=metadataClm
)

figClm.show()

figCredit = get_column_plot(
    real_data=creditDataTrain,
    synthetic_data=syntheticCreditData,
    column_name='Class',
    metadata=metadataCredit
)

figCredit.show()