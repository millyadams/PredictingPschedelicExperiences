#project_functions

#functions for project 


import numpy as np 
import pandas as pd 

#For plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Train/test 
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

#Eval metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report,  precision_recall_fscore_support, make_scorer 

#Models 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 

#Feature selection 
from sklearn.feature_selection import RFECV

#Feature importance
import shap 

#save best model 
from joblib import dump
import json
import os


#Load data 
def load_data(filepath):
    """Load data"""
    data = pd.read_csv(filepath)
    return data 
    
#Preprocessing steps - train/test and scale 
def preprocess_data(X, y, test_size = 0.2, random_state = 42):
    """
    Split data into train/test 
    Scale feature data (X) 
    Return split and scaled datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = random_state, shuffle = True, stratify=y) 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test 

def confidence_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    h = std_err * 1.96  # For 95% confidence interval
    return mean - h, mean + h

#Feature selection 
def feature_selection(X_train, X_test, y_train, model, cv):
    """Employ RFECV to model and store model-specific reduced feature set 

    Args:
        X_train: training features
        y_train: training labels 
        model: classifier 
        cv: cross validation folds

    Returns:
        X_train_SF: training data with reduced feature set 
        X_test_SF: test data with reduced feature set 
        rfecv.support_: Boolean array of selected features 
        rfecv: fitted RFECV object 
        optimal_features: number of optimal features 
    """
    rfecv = RFECV(estimator = model, step = 1, cv=cv, scoring = 'accuracy') #rfecv initialise with given parameters 
    rfecv.fit(X_train, y_train) #rfecv model fitted to training data to find optimal number of features 
    X_train_SF = rfecv.transform(X_train) #transform training data to get reduced feature set 
    X_test_SF = rfecv.transform(X_test)
    optimal_features = rfecv.n_features_
    return X_train_SF, X_test_SF, rfecv.support_, rfecv, optimal_features

#feature selection plotting 
def plot_rfecv(rfecv, model_name, optimal_features, save_dir):
    """
    Plots number of features vs. cross validation scores and saves the figure
    Creates and saves plot that visualises the relationship between number of features 
    selected and cross-validation accuracy score 

    Args:
    - rfecv: Fitted RFECV instance 
    - model_name: Name of the model 
    - optimal_features: Number of optimal features identified by RFECV 
    - save_dir: Directory where the plot should be save 
    
    """
    
    plt.figure(figsize=(10, 8)) #create plot 
    plt.title(f"Feature selection via Cross Validation for {name}")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross-validation score (accuracy)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score']) #plot range of all features mean test score (each ft has cv test score)
    
    #Plot optimal number of features in the figure 
    plt.axvline(optimal_features, color='r', linestyle='--')  #Add a vertical line at the optimal number of features
    plt.figtext(0.1, -0.05, f"Optimal number of features: {optimal_features}", fontsize=12, ha='left', color='black')
    #Save figure 
    save_path = f"{save_dir}/{model_name}_feature_selection.png"
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close() 
    
    
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, f1, precision, recall, cm 
    
def plot_cm(cm, class_names, name, savedir):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot = True, cmap='Blues', xticklabels = class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.title(f'Confusion Matrix for {name}', fontsize = 16)
    plt.savefig(f'{savedir}_{name}_confusion_matrix.png', bbox_inches ='tight')
    plt.close() 
    
    
def shap_analysis(model, X_train, X_test, name, feature_names, savedir):
        
    name = name.replace('_', ' ')

    #select SHAP explainer based on type od model 
    #if isinstance(model,(RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
        #explainer = shap.TreeExplainer(model) #different explainers for different model types 
    #else: 
        #explainer = shap.KernelExplainer(model.predict, X_train)
        
    #shap_values = explainer.shap_values(X_test) #calculate shap values for test data using explainer 
    #on test dataset reflects performance and feature importance on unseen data 
    
        # select SHAP explainer based on the type of model
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)):
        if isinstance(model, XGBClassifier):
            # Check if XGBoost uses 'gbtree' booster
            booster_type = model.get_params().get('booster', 'gbtree')  # Default to 'gbtree' if not specified
            if booster_type == 'gbtree':
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_train)  # Use LinearExplainer for gblinear
        else:
            explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train)
        
    shap_values = explainer.shap_values(X_test)  # calculate shap values for test data using explainer
    
    #Summary plot 
    plt.figure(figsize=(10,7), dpi = 300)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type = "bar", show = False)
    plt.title(f"SHAP Feature Importance for {name}")
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=10)
    plt.savefig(f'{savedir}/{name}_shap_summary.png', bbox_inches='tight')
 
        
    #Detailed plot 
    plt.figure(figsize = (10, 7), dpi = 300) 
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show = False) 
    plt.title(f"SHAP Feature Importance for {name}", fontsize = 14, y = 1.08)
    plt.xlabel("SHAP value (impact on model output)", fontsize=10)
    plt.savefig(f'{savedir}/{name}_shap_detailed.png', bbox_inches='tight')
        
