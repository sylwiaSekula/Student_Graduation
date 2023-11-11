import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from scripts.prepare_and_train_models import *


def load_model(model_dir: str, model_file: str) -> object:
    """
    Load a machine learning model from a pickle file
    :param model_dir: (str), the directory where the model file is located
    :param model_file: (str), the filename of the pickled machine learning model.
    :return: a model loaded from the specified file.
    """
    return pickle.load(open(os.path.join(model_dir, model_file), 'rb'))


def plot_confusion_matrix(matrix: np.ndarray, classes: list, filename: str) -> plt.Figure:
    """
    Plot a confusion matrix as a heatmap.
    :param matrix: (np.ndarray)
    :param classes: (list of str), the list of class labels
    :param filename: (str), optional, if provided the plot will be saved to a file in the "plots" folder.
    :return: The confusion matrix plot
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{filename}')

    if filename:
        plt.savefig(f'plots/{filename}.png', format='png')

    plt.show()

def main():
    # load the test dataset
    df_test = pd.read_csv('df_test.csv')
    # define the target and features
    target = 'Target'
    X_test = df_test.drop(target, axis=1)  # features
    y_test = df_test[target]  # target
    # load the trained feature selectors
    sel_log = load_model(trained_model_dir, selection_file_log)
    sel_svm = load_model(trained_model_dir, selection_file_svc)
    sel_lgb = load_model(trained_model_dir, selection_file_lgb)
    # load the trained label encoder
    le = load_model(trained_model_dir, le_file)
    scaler = load_model(trained_model_dir, scaler_file)
    # load the trained models
    log = load_model(trained_model_dir, log_file)
    svm = load_model(trained_model_dir, svc_file)
    lgb = load_model(trained_model_dir, lgb_file)

    # encode the test target using the trained label encoder
    y_test = le.transform(y_test)
    # scale the test data
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
    # transform the test data using the trained feature selectors
    X_test_pipe = sel_log.transform(X_test)
    X_test_svm = sel_svm.transform(X_test)
    X_test_lgb = sel_lgb.transform(X_test)
    # predict the target on the test data using the trained models
    y_pred_log = log.predict(X_test_pipe)
    y_pred_svm = svm.predict(X_test_svm)
    y_pred_lgb = lgb.predict(X_test_lgb)

    # print the classification report for each model
    predictions = [('logistic regression', y_pred_log), ('Support Vector Classifier', y_pred_svm),
                   ('Light Gradient Boosting Machine', y_pred_lgb)]
    encoded_classes = le.classes_

    for prediction_name, prediction in predictions:
        report = classification_report(y_test, prediction, target_names=encoded_classes)
        matrix = confusion_matrix(y_test, prediction)
        print(f'{prediction_name}')
        print(report)
        plot_confusion_matrix(matrix, encoded_classes, filename=f'{prediction_name}_confusion_matrix')


if __name__ == '__main__':
    main()
