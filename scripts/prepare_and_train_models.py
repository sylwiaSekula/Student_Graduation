import os
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.settings import *


def train_and_save_model(model: object, X_train: pd.DataFrame, y_train: pd.DataFrame, trained_model_dir: str,
                         selection_file: str, model_file: str, n_features_to_select: int, direction: str) -> None:
    """
    Fit a SequentialFeatureSelector on the train dataset, select the specified number of features, and then fits the
    model on the transformed train dataset. Save both the feature selector and the model.
    :param model: the machine learning model
    :param X_train: pd.DataFrame, the train dataset with the features
    :param y_train: pd.DataFrame, the train dataset with the target
    :param trained_model_dir: str, the directory where trained models and selectors will be saved
    :param selection_file: str, the filename for the saved feature selector
    :param model_file: str, the filename for the saved machine learning model
    :param n_features_to_select: int, the number of features to select during feature selection
    :param direction: str, the direction of the sequential feature selection
    :return: None
    """
    sfs_forward = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction)
    sfs_forward.fit(X_train, y_train)
    X_train_transformed = sfs_forward.transform(X_train)
    model.fit(X_train_transformed, y_train)
    pickle.dump(sfs_forward, open(os.path.join(trained_model_dir, selection_file), 'wb'))
    pickle.dump(model, open(os.path.join(trained_model_dir, model_file), 'wb'))


def evaluate_model(model: object, X_train: pd.DataFrame, y_train: pd.DataFrame) -> float:
    """
    Evaluate the machine learning model using cross-validation score
    :param model:
    :param X_train: pd.DataFrame, the train dataset with the features
    :param y_train: pd.DataFrame, the train dataset with the target
    :return:float, the mean F1 macro score obtained through 10-fold cross-validation
    """
    f1_macro_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='f1_macro')
    return np.mean(f1_macro_scores)


def main():
    # get the current directory
    current_directory = os.getcwd()
    # set input and output filenames and paths
    input_filename = 'graduation_dataset.csv'
    output_test_filename = 'df_test.csv'
    input_path = os.path.join(current_directory, input_filename)
    output_test_path = os.path.join(current_directory, output_test_filename)
    # define the hyperparameters
    n_features_to_select = 20
    direction = "forward"
    random_state = 42
    test_size = 0.33
    # load the dataset
    df = pd.read_csv(input_path)
    # define the target and the features
    target = 'Target'
    # filter the dataset to exclude rows 'Enrolled' in the 'target' column
    df = df[df[target] != 'Enrolled']
    X = df.drop(target, axis=1)  # features
    y = df[target]  # target

    # split the dataset into the train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # concat the test target and features and save the test dataset to a CSV file
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test.to_csv(output_test_path, index=False)

    # encode the target labels and save the trained LabelEncoder
    le = preprocessing.LabelEncoder()
    y_train_resampled = le.fit_transform(y_train)
    pickle.dump(le, open(os.path.join(trained_model_dir, le_file), 'wb'))
    # scale the data in numerical columns and save the trained Standardscaler
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    pickle.dump(scaler, open(os.path.join(trained_model_dir, scaler_file), 'wb'))
    # create machine learning models and pipelines
    logistic = LogisticRegression(random_state=random_state, class_weight="balanced")
    svc = SVC(random_state=random_state, class_weight='balanced')
    lgb = LGBMClassifier(random_state=random_state)

    models_and_files = [
        ("Logistic Regression", logistic, selection_file_log, log_file),
        ("Support Vector Classifier", svc, selection_file_svc, svc_file),
        ("LightGBM Classifier", lgb, selection_file_lgb, lgb_file),
    ]

    # fit each model on the training data, save feature selector and model
    for model_name, model, selection_file, model_file in models_and_files:
        train_and_save_model(model, X_train, y_train_resampled, trained_model_dir, selection_file, model_file,
                             n_features_to_select, direction)
        mean_f1_macro = evaluate_model(model, X_train, y_train_resampled)
        print(f'{model_name}: {mean_f1_macro}')


if __name__ == '__main__':
    main()
