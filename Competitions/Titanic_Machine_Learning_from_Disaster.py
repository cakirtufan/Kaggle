# -*- coding: utf-8 -*-
"""
Titanic Survival Prediction with Feature Engineering (Class-based)
Created on Sat Sep 20 01:04:57 2025
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


class TitanicModel:
    def __init__(self, train_path, test_path, gender_submission_path, model=None):
        self.train_path = train_path
        self.test_path = test_path
        self.gender_submission_path = gender_submission_path

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
        self.gender_df = pd.read_csv(self.gender_submission_path)

    def feature_engineering(self):
        # 1. Age Groups
        bins = [0, 3, 12, 19, 35, 60, 80]
        labels = ["Baby", "Child", "Teen", "Adult", "Senior", "Elder"]
        self.train_df["AgeGroup"] = pd.cut(self.train_df["Age"], bins=bins, labels=labels, right=True)
        self.test_df["AgeGroup"] = pd.cut(self.test_df["Age"], bins=bins, labels=labels, right=True)

        # 2. Family size
        self.train_df["FamilySize"] = self.train_df["SibSp"] + self.train_df["Parch"] + 1
        self.train_df["IsAlone"] = (self.train_df["FamilySize"] == 1).astype(int)

        self.test_df["FamilySize"] = self.test_df["SibSp"] + self.test_df["Parch"] + 1
        self.test_df["IsAlone"] = (self.test_df["FamilySize"] == 1).astype(int)

        # 3. Fare binning
        self.train_df["FareBin"] = pd.qcut(self.train_df["Fare"], 3, labels=["Low", "Medium", "High"])
        self.test_df["FareBin"] = pd.qcut(self.test_df["Fare"], 3, labels=["Low", "Medium", "High"])

        # 4. Cabin letter
        self.train_df["CabinLetter"] = self.train_df["Cabin"].astype(str).str[0].replace("n", "U")
        self.test_df["CabinLetter"] = self.test_df["Cabin"].astype(str).str[0].replace("n", "U")

        # 5. Embarked fill
        self.train_df["Embarked"] = self.train_df["Embarked"].fillna("S")
        self.test_df["Embarked"] = self.test_df["Embarked"].fillna("S")

    def prepare_data(self):
        features = ["Sex", "Pclass", "AgeGroup", "FamilySize", "IsAlone",
                    "FareBin", "CabinLetter"]

        self.X = self.train_df[features].copy()
        self.y = self.train_df["Survived"]
        self.X_val = self.test_df[features].copy()
        self.y_val = self.gender_df["Survived"]

        # Encode categorical
        self.X["Sex"] = self.X["Sex"].map({"male": 0, "female": 1})
        self.X_val["Sex"] = self.X_val["Sex"].map({"male": 0, "female": 1})

        # One-hot encode categorical
        categorical_cols = ["AgeGroup", "FareBin", "CabinLetter"]
        self.X = pd.get_dummies(self.X, columns=categorical_cols, drop_first=True)
        self.X_val = pd.get_dummies(self.X_val, columns=categorical_cols, drop_first=True)

        # Align validation
        self.X_val = self.X_val.reindex(columns=self.X.columns, fill_value=0)

    def train(self):
        self.model.fit(self.X, self.y)

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_val)
        acc = accuracy_score(self.y_val, self.y_pred)
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring="accuracy")
        print("Accuracy (on gender_submission):", acc)
        print("Cross-validation accuracy:", cv_scores.mean())

    def feature_importance(self, plot=True):
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10,
                                        random_state=42, n_jobs=-1)
        importances = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)
        print(importances)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.barh(importances["Feature"], importances["Importance"])
            plt.gca().invert_yaxis()
            plt.title("Feature Importance (Permutation)")
            plt.show()

    def save_submission(self, filename="titanic_submission.csv"):
        submission = pd.DataFrame({
            "PassengerId": self.test_df["PassengerId"],
            "Survived": self.y_pred
        })
        submission.to_csv(filename, index=False)
        print(f"Submission file '{filename}' created!")


if __name__ == "__main__":
    os.chdir(r"C:\Users\cakir")

    train_path = r"Downloads\titanic\train.csv"
    test_path = r"Downloads\titanic\test.csv"
    gender_submission_path = r"Downloads\titanic\gender_submission.csv"

    # Initialize with SVC
    tm = TitanicModel(train_path, test_path, gender_submission_path, model=SVC())

    tm.load_data()
    tm.feature_engineering()
    tm.prepare_data()
    tm.train()
    tm.evaluate()
    tm.feature_importance()
    tm.save_submission()
