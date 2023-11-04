from sklearn.metrics import classification_report

from scripts.prepare_and_train_models import *


def load_model(model_dir: str, model_file: str) -> object:
    """
    Load a machine learning model from a pickle file
    :param model_dir: (str), the directory where the model file is located
    :param model_file: (str), the filename of the pickled machine learning model.
    :return: a model loaded from the specified file.
    """
    return pickle.load(open(os.path.join(model_dir, model_file), 'rb'))


def main():
    # load the test dataset
    df_test = pd.read_csv('df_test.csv')
    # define the target and features
    target = 'Target'
    X_test = df_test.drop(target, axis=1)  # features
    y_test = df_test[target]  # target
    # load the trained feature selectors
    sel_pipe = load_model(trained_model_dir, selection_file_log)
    sel_svm = load_model(trained_model_dir, selection_file_svc)
    sel_lgb = load_model(trained_model_dir, selection_file_lgb)
    # load the trained label encoder
    le = load_model(trained_model_dir, le_file)
    # load the trained models
    pipe = load_model(trained_model_dir, pipe_log_file)
    svm = load_model(trained_model_dir, svc_pipe_file)
    lgb = load_model(trained_model_dir, lgb_file)

    # encode the test target using the trained label encoder
    y_test = le.transform(y_test)
    # transform the test data using the trained feature selectors
    X_test_pipe = sel_pipe.transform(X_test)
    X_test_svm = sel_svm.transform(X_test)
    X_test_lgb = sel_lgb.transform(X_test)
    # predict the target on the test data using the trained models
    y_pred_log = pipe.predict(X_test_pipe)
    y_pred_svm = svm.predict(X_test_svm)
    y_pred_lgb = lgb.predict(X_test_lgb)

    # print the classification report for each model
    predictions = [('logistic regression', y_pred_log), ('svm', y_pred_svm), ('lgb', y_pred_lgb)]
    encoded_classes = le.classes_

    for prediction_name, prediction in predictions:
        report = classification_report(y_test, prediction, target_names=encoded_classes)
        print(f'{prediction_name}')
        print(report)


if __name__ == '__main__':
    main()
