# GenAI_Insurance_Finance
Demo for CTGAN implementations in insurance and finance industry 

With increasing popularity of Generative AI, generative adversarial network (GAN)/conditional GAN (ctGAN) are attracting attention from traditional industries like finance and insurance (BCG 2023a, BCG 2023b, Kuo 2019), where they can be used in generating synthetic data for training machine learning models. GAN/ct GAN were early more used for generating image data, where pixels' values follow a Gaussian-like distribution. Compared with image data, tabular data includes numerical and categorical columns, where numerical columns generally do not follow a Gaussian distribution, which brings more challenges in training the synthesizer model. Moreover, multiple models in numerical columns and highly imbalanced categorical columns are making more challenges (Xu et al. 2019). 

To resolve the challenges in building GAN model for generating tabular data, Xu et al. (2019) proposed the use of mode-specific normalization method to process the non-Gaussian and multimodal distribution, and conditional generator and training-by-sampling to handle the imbalanced distribution of categorical columns. Here, based on Xu et al. (2019), I created two demonstrations of how to train a ctGAN synthesizer, how to use the trained synthesizer to generate synthetic data, and how to evaluate synthesizer quality based on two metrics: 1) whether columns in synthetic data follow same joint distribution as training data?; and 2) whether the machine learning models trained based on synthetic and (original) training data have similar performance on the test dataset?

## Data 
Here used two public datasets from Kaggle: 1) French Third-Party Liability claim frequency dataset (https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims?select=freMTPL2freq.csv); and 2) Credit Card Fraud Detection dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv).

## Code
Four python files were included in this repository (src).
1. ctGANtrain.py: This script mainly focuses on the training of synthesizer models with claim and credit card datasets. Before the training, both datasets were split into training and test data, as later test data will be used for evaluating the performance of synthesizer models. To be noted, users need to adjust the column types detected by the detect_from_dataframe function, particularly when there is id information (e.g., 'IDpol' in the claim dataset).
2. ctGANdataGeneration.py: This script is used for generating synthetic data with trained synthesizers. Things to be noted is the current sdv package cannot process multiple in-parallel instances efficiently, users have to defined distinct temporary paths for each instance and assign them to the 'output_file_path' parameter, which are covered in this script.
3. ctGANdataEvaluation_1.py: As evaluation criteria 1 of trained synthesizer, this script helps generate overall quality scores of both synthesizer models, as well as evaluation properties including column shapes and pair trends. Additionally, users can use the visualization functions to illustrate the comparisons of particular columns from original and synthetic data.
4. ctGANdataEvaluation_2.py: As evaluation criteria 2 of trained synthesizer, this script trained machine learning models on both original training and synthetic data with CatBoost for claim and credit card respectively, and then tested their performances on the held-out test datasets in ctGANtrain.py. Comparing the machine learning models (trained based on original training and synthetic data respectively) performance metrics on the same claim (credit card) test dataset, can help us assess the efficiency of trained synthesizers.

## Sample Results
In the results file, I have included some sample results figures/screenshots.

## Requirements
The requirements.txt file specifies the package versions used in the scripts.

## References
1. BCG (2023). Leading Insurers Are Having a Generative AI Moment. <https://www.bcg.com/publications/2023/why-insurance-leaders-need-to-leverage-gen-ai>, last accessed on Sep 16, 2023.
2. BCG (2023). Generative AI in the Finance Function of the Future. <https://www.bcg.com/publications/2023/generative-ai-in-finance-and-accounting>, last accessed on Sep 16, 2023.
3. Kuo, K. (2019). Generative synthesis of insurance datasets. arXiv preprint arXiv:1912.02423.
4. Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional gan. Advances in neural information processing systems, 32.
