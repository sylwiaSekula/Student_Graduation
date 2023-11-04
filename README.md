# Student Graduation Classification Project

This project focuses on a classification task using a dataset from Kaggle, which can be accessed here:https://www.kaggle.com/datasets/ranzeet013/student-graduation-dataset. 
The dataset contains information about students, including various features that can be used to predict their graduation status.The target variable in this dataset originally had three unique values: "Dropout," "Graduate," and "Enrolled." However, "Enrolled" was excluded from the target column since it represents students who are still studying.


## Project Overview
The target data was encoded using Label Encoder to convert the categorical labels into numerical values suitable for machine learning algorithms. Additionally, I performed feature selection for each model using the Sequential Feature Selector from the scikit-learn library, selecting 20 relevant features using forward selection.
Then, I used three different machine learning models to predict the students' graduation status. The models used are:
1. Logistic Regression:
    Logistic Regression estimates the probability of a binary outcome, making it a simple and effective classification method.
2. Support Vector Classifier (SVC):
   Support Vector Classifier (SVC) finds a hyperplane that best separates classes in the data, using support vectors to define the decision boundary.
3. Light Gradient Boosting Machine (LGBM):
        Light Gradient Boosting Machine (LGBM) is an efficient ensemble learning algorithm that builds decision trees sequentially to minimize prediction errors and handle diverse datasets effectively.
I saved all the fitted models, Sequential Feature Selector objects, and Label Encoders so that they can be loaded and reused for future testing on new data.

## Model Evaluation

The performance of each model was evaluated on a test dataset, and I got the following classification metrics:
Logistic Regression
              precision    recall  f1-score   support

     Dropout       0.90      0.85      0.88       467
    Graduate       0.91      0.94      0.92       731

    accuracy                           0.91      1198
    macro avg      0.91      0.90      0.90      1198
    weighted avg   0.91      0.91      0.91      1198

Support Vector Classifier 
              precision    recall  f1-score   support

     Dropout       0.90      0.84      0.87       467
    Graduate       0.90      0.94      0.92       731

    accuracy                           0.90      1198
    macro avg      0.90      0.89      0.90      1198
    weighted avg   0.90      0.90      0.90      1198

Light Gradient Boosting Machine
              precision    recall  f1-score   support

     Dropout       0.93      0.84      0.88       467
    Graduate       0.90      0.96      0.93       731

    accuracy                           0.91      1198
    macro avg      0.92      0.90      0.91      1198
    weighted avg   0.91      0.91      0.91      1198

Conclusions:
All three models demonstrated good overall performance with high precision, recall, F1-scores, and accuracy. The results suggest that the machine learning models applied in this project are well-suited for the task of predicting student graduation status.

