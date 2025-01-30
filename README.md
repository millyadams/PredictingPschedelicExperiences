# PredictingPschedelicExperiences
## Investigating EEG Predictors of the Acute DMT Experience
This repository contains code for the analyses of my MSc Translational Neuroscience and subsequent paper (in-prep) investigating whether features of participants' baseline, resting-state EEG signals can be used to predict binarised classifications of subsequent DMT experiences (e.g., Mystical vs Non-Mystical). 

## Dependencies 
To run this code, you will need the following dependencies in your environment:
- python 3.11.5
- MATLAB_R2024b
- pandas 2.0.3
- numpy 1.24.3

## Datasets 
The pipeline uses EEG and subjective experience data from 50 healthy control (HC) participants across three studies collected by the Centre for Pyschedelic Research at Imperial College London: [Timmerman et al., 2019](https://www.nature.com/articles/s41598-019-51974-4), [Timmerman et al., 2023](https://www.pnas.org/doi/10.1073/pnas.2218949120), [Luan et al., 2024](https://journals.sagepub.com/doi/10.1177/02698811231196877). 

### EEG Data
EEG features are extracted from resting-state EEG recordings collected immediately prior to DMT intraveneous infusion.
### Subjective Experience Data 
Three experience classifications are used as binary outcome labels: Mystical, Visual and Anxious. 

## Pipeline 
Dataframe: matrix containing binary experience labels (3) and EEG features (448) for 50 participants, in the case of our data. 

Multiple pipelines are run and are differentiated by: 
1. Type of experience - predicting binary labels for Mystical, Visual and Anxious experiences separately
2. Type of EEG Power - we wanted to compare models built using both relative and absolute power

### Steps 
Because we aimed to identify the optimal classifier to use to solve this problem, we compared performance of 7 classifiers. The below figure outlines pipeline stages. 
![Screenshot 2024-11-04 at 14 19 17](https://github.com/user-attachments/assets/9bc15cbc-0e0e-4c24-affd-9c92eb0633a2)
1. Train/Test Split
2. Dimensionality reduction of the feature space using recursive feature elimination with cross-validation (RFECV) to create model-specific optimal feature sets
3. Hyperparameter tuning using random search
4. Test each model on unseen data (note - each model has a personalised feature set and optimised hyperparameteres)
5. Feature Importance analysis using SHapely Additive exPlanations analysis (SHAP) 

