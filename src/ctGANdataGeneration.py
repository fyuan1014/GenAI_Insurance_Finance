import pandas as pd
from sdv.single_table import CTGANSynthesizer
import random
import tempfile
import os

import warnings
warnings.filterwarnings('ignore')

### 1. Loading synthesizers
#### Synthesizer models were read from your saved path defined in ctGANtrain.py

serializedMyModelClm = CTGANSynthesizer.load('.../ctganClmModel.pkl')
serializedMyModelCredit = CTGANSynthesizer.load('.../ctganCreditModel.pkl')

### 2. Checking synthesizer parameters
#serializedMyModelClm.get_parameters()
#serializedMyModelCredit.get_parameters()

### 3. Generating synthetic data
claimFrequencyDataTrain=pd.read_csv(".../freMTPL2freq_train.csv")
creditDataTrain=pd.read_csv(".../creditcard_train.csv")

#### 3.1 Generating claim symthetic data
temp_dir = tempfile.mkdtemp()
dataSourceNum = os.getpid()
tempPath = os.path.join(temp_dir, f"temp.{dataSourceNum}.csv")
syntheticClmData = serializedMyModelClm.sample(num_rows=len(claimFrequencyDataTrain),output_file_path=tempPath)
if os.path.exists(tempPath):
    os.remove(tempPath)
    
#### 3.2 Generating credit symthetic data
temp_dir = tempfile.mkdtemp()
dataSourceNum = os.getpid()
tempPath = os.path.join(temp_dir, f"temp.{dataSourceNum}.csv")
syntheticCreditData = serializedMyModelCredit.sample(num_rows=len(creditDataTrain),output_file_path=tempPath)
if os.path.exists(tempPath):
    os.remove(tempPath)

### 4. Saving generated synthetic data
syntheticClmData.to_csv(".../freMTPL2freq_train_synthetic.csv")
syntheticCreditData.to_csv(".../creditcard_train_synthetic.csv")