# Multi-Class Logistic Regression for Incident Classification

## Overview
This Jupyter Notebook performs multi-class classification using logistic regression on an incident dataset. The goal is to predict the `state` of an incident based on features such as `priority`, `ai_category`, `issue_type`, `device_name`, and `account_id`.

## Dataset
The dataset consists of incident records with the following columns:

- `priority`: The priority level of the incident (e.g., "1 - Critical", "2 - High").
- `ai_category`: The AI-determined category of the issue.
- `issue_type`: A description of the issue type.
- `device_name`: The type of device involved in the incident.
- `account_id`: An anonymized account identifier.
- `state`: The target variable representing the incident's current status.

## Preprocessing Steps
- Missing values in `device_name` were replaced with `"unknown"`.
- Missing values in `issue_type` were replaced with `"Other"`.
- `account_id` values were anonymized.
- The dataset index was reset to standard numbering.
- Categorical variables were encoded for model compatibility.

## Model
A multi-class logistic regression model was trained using scikit-learn. The model was evaluated using:

- **Accuracy Score**: Measures overall classification performance.

## Results
- The logistic regression model achieved an accuracy score of `90%`.
- Performance varied across different incident categories.


