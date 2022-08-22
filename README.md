This is the code for the approval odds model on https://ccapprovalodds.com.

Most of it is in Jupyter notebooks, but the main classes for training and prediction are in `csr.py`.

The dataset is exported as a CSV from the Raw Data tab in this spreadsheet: https://docs.google.com/spreadsheets/d/1gMQn4ZMvtFA125BuED0l1qB4u1vplmMc3JvQtcURC-o/edit?usp=sharing

in `xgb-training.ipynb`, a grid search is conducted for a gradient boosted decision trees model.

Currently, there is only a model for the instant approval, pending, and denial odds, but one could be trained for the credit limit.

In `dataset-analysis`, graphs are generated for the distribution, approval percentages, and the model's opinion on various fields. 
