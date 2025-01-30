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
<img width="586" alt="Screenshot 2025-01-30 at 21 26 07" src="https://github.com/user-attachments/assets/af2549b1-b3f5-46ce-ab5a-a65399703775" />

