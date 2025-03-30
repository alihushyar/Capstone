# Analysis of Incident Tickets and Predicting Resolution Queue

## Overview
This Jupyter Notebook (`Incident_Analysis_v1.0.ipynb`) contains an analysis of 76K records of incident data, including data preprocessing, exploratory data analysis (EDA), and machine learning modeling. The primary objective is to predict the state of an incident using various classification techniques.  This type of prediction can help in better triage and prioritization of alerts.  

## Dataset
The dataset consists of incident records with the following columns:

- `priority`: The priority level of the incident (e.g., "1 - Critical", "2 - High").
- `ai_category`: The AI-determined category of the issue.
- `issue_type`: A description of the issue type.
- `device_name`: The type of device involved in the incident.
- `account_id`: An anonymized account identifier.
- `state`: The target variable representing the incident's current status.

Ticket States
0	RNC	 - Resolved Tier 1
1	CNC	 - Cancelled Tieir 1
2	RWC	- Resolved Tier 2
3	HNC	- On Hold Tier 1
4	HWC	- On Hold Tier 2
5	Closed
6	CWC	- Cancelled Tier 2

## Preprocessing Steps
- Missing values in `device_name` were replaced with `"unknown"`.
- Missing values in `issue_type` were replaced with `"Other"`.
- `account_id` values were anonymized.
- The dataset index was reset to standard numbering.
- Categorical variables were encoded for model compatibility.

## Features

- **Data Cleaning**: Reads and preprocesses incident data from a CSV file.
- **Exploratory Data Analysis (EDA)**: Uses Seaborn, Plotly, and Matplotlib for visualizations.
- **Feature Engineering**: Handles categorical data.
- **Baseline Model**: Implements a multi-class logistic regression model for comparison.
- **Machine Learning Models**: Trains and evaluates ensemble models (*Random Forest, XGBoost, SVC, Gradient Boosting, Stacking Classifier, Neural Network*,).

Decision Tree Results:
Train Time: 0.1830 seconds
Accuracy: 0.5992
              precision    recall  f1-score   support

         CNC       0.99      0.81      0.89      4130
         CWC       0.04      0.50      0.08       133
      Closed       0.08      0.79      0.14       168
         HNC       0.91      0.91      0.91      1501
         HWC       0.13      0.51      0.21       403
         RNC       0.79      0.57      0.66      5253
         RWC       0.59      0.26      0.36      3501

    accuracy                           0.60     15089
   macro avg       0.50      0.62      0.47     15089
weighted avg       0.78      0.60      0.66     15089


Random Forest Results:
Best Parameters: {'class_weight': 'balanced', 'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
Train Time: 106.3309 seconds
Accuracy: 0.6140
              precision    recall  f1-score   support

         CNC       0.99      0.81      0.89      4130
         CWC       0.04      0.47      0.08       133
      Closed       0.08      0.82      0.14       168
         HNC       0.93      0.90      0.92      1501
         HWC       0.16      0.48      0.24       403
         RNC       0.76      0.64      0.69      5253
         RWC       0.61      0.24      0.34      3501

    accuracy                           0.61     15089
   macro avg       0.51      0.62      0.47     15089
weighted avg       0.77      0.61      0.66     15089

Multinomial Logistic GridSearch
Fitting 5 folds for each of 5 candidates, totalling 25 fits
Best Parameters: {'C': 100}
Accuracy: 0.5890
              precision    recall  f1-score   support

         CNC       0.98      0.80      0.88      4130
         CWC       0.04      0.47      0.07       133
      Closed       0.08      0.81      0.14       168
         HNC       0.87      0.90      0.89      1501
         HWC       0.13      0.41      0.20       403
         RNC       0.75      0.59      0.66      5253
         RWC       0.57      0.21      0.31      3501

    accuracy                           0.59     15089
   macro avg       0.49      0.60      0.45     15089
weighted avg       0.75      0.59      0.64     15089

Stacking Random Forest with XGBoost
Accuracy: 0.7714
              precision    recall  f1-score   support

         CNC       0.99      0.82      0.89      4130
         CWC       0.00      0.00      0.00       133
      Closed       0.26      0.05      0.09       168
         HNC       0.97      0.89      0.93      1501
         HWC       0.44      0.10      0.16       403
         RNC       0.69      0.90      0.78      5253
         RWC       0.65      0.62      0.63      3501

    accuracy                           0.77     15089
   macro avg       0.57      0.48      0.50     15089
weighted avg       0.77      0.77      0.76     15089

Neural Network Accuracy: 0.5774
              precision    recall  f1-score   support

         CNC       0.98      0.78      0.87      4124
         CWC       0.04      0.51      0.07       128
      Closed       0.07      0.82      0.13       174
         HNC       0.90      0.92      0.91      1501
         HWC       0.12      0.42      0.19       377
         RNC       0.74      0.55      0.63      5244
         RWC       0.58      0.24      0.34      3541

    accuracy                           0.58     15089
   macro avg       0.49      0.61      0.45     15089
weighted avg       0.76      0.58      0.63     15089

## Results

Summary of Metrics:
Overall Accuracy: 77.14%. This means the model correctly predicted 77.14% of the instances in your dataset. This is a relatively good overall performance but masks significant variation across classes.

Class-Level Performance:
1. CNC (Class 1):
Precision: 0.99 (Very High) - When the model predicts CNC, it is almost always correct.

Recall: 0.82 (High) - The model correctly identifies 82% of all actual CNC instances.

Support: 4130 instances.

This class performs very well, indicating the model has learned this class well.

2. CWC (Class 2):
Precision: 0.00 (Very Poor) - The model does not correctly predict any instances of this class.

Recall: 0.00 (Very Poor) - It misses all actual instances of this class.

Support: 133 instances.

Possible Issue: The model is not detecting this class at all. This could be due to class imbalance, insufficient data, or features that do not help distinguish this class.

3. Closed (Class 3):
Precision: 0.26 (Low) - The model is only partially correct when predicting Closed.

Recall: 0.05 (Very Low) - The model misses most of the true Closed instances.

Support: 168 instances.

Possible Issue: This class is underrepresented in the dataset, or the model is struggling to distinguish it. Exploring oversampling or different features might help.

4. HNC (Class 4):
Precision: 0.97 (Very High) - The model is very accurate when predicting HNC.

Recall: 0.89 (High) - It also captures most of the HNC instances.

Support: 1501 instances.

This class is well-represented and the model performs well here.

5. HWC (Class 5):
Precision: 0.44 (Moderate) - Somewhat accurate predictions.

Recall: 0.10 (Low) - The model misses most HWC instances.

Support: 403 instances.

Possible Issue: This class is underperforming due to imbalanced class distribution or lack of distinct features for differentiation.

6. RNC (Class 6):
Precision: 0.69 (Moderate) - Reasonably good precision.

Recall: 0.90 (Very High) - The model correctly identifies most RNC instances.

Support: 5253 instances.

This class is performing well, especially in terms of recall.

7. RWC (Class 7):
Precision: 0.65 (Moderate) - Somewhat accurate predictions.

Recall: 0.62 (Moderate) - The model correctly identifies a decent number of RWC instances.

Support: 3501 instances.

This class has a fair performance, but there’s room for improvement.


## Key Takeaways:
The model performs very well on certain classes (CNC, HNC, RNC) but poorly on others (CWC, Closed, HWC).

Class Imbalance: There might be a class imbalance issue, particularly for classes like CWC, Closed, and HWC, which have relatively low precision and recall. Techniques like oversampling, undersampling, or class weighting may help address this.

Feature Engineering: It might be worth exploring different features or transforming the data to improve the model’s ability to distinguish difficult classes.

Model Improvements: You could consider tuning the model or trying other models that handle imbalanced data better, such as Random Forest, Gradient Boosting, or using neural networks with class weights.
