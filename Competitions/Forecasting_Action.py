# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 02:13:28 2025

@author: cakir
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance


class ForcastingAction:
    def __init__(self, train_path, test_path, gender_submission_path, model=None):
        self.train_path = train_path
        self.test_path = test_path
        self.sample_submission_path = sample_submission_path

        # Default model: SVC if none is provided
        self.model = model if model else SVC(probability=True)

        # Data placeholders
        self.train_df = None
        self.test_df = None
        self.gender_df = None
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.y_pred = None

    def load_data(self):
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.sample_df = pd.read_csv(self.sample_submission_path)
        
if __name__ == "__main__":
    os.chdir(r"C:\Users\cakir")

    train_path = r"Downloads\forecastin_for_action\train.csv"
    test_path = r"Downloads\forecastin_for_action\test.csv"
    sample_submission_path = r"Downloads\forecastin_for_action\sample_submission.csv"
    
    ForcastingAction = ForcastingAction(train_path, test_path, sample_submission_path)
