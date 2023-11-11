# Student Graduation Classification Project

This project focuses on a classification task using a dataset from Kaggle, which can be accessed here:https://www.kaggle.com/datasets/ranzeet013/student-graduation-dataset. 
The dataset contains information about students, including various features that can be used to predict their graduation status. The dataset has 35 columns and 4424 entries.
The columns in the dataset:
1. Marital status	- The marital status of the student (Categorical)
2. Application mode	- The method of application used by the student (Categorical)
3. Application orde - The order in which the student applied (Numerical)
4. Course - The course taken by the student (Categorical)
5. Daytime/evening attendance- Whether the student attends classes during the day or in the evening (Categorical)
6. Previous qualification -The qualification obtained by the student before enrolling in higher education (Categorical)
7. Nacionality - The nationality of the student (Categorical)
8. Mother's qualification - The qualification of the student's mother (Categorical)
9. Father's qualification - The qualification of the student's father (Categorical)
10. Mother's occupation - The occupation of the student's mother (Categorical)
11. Father's occupation	- The occupation of the student's father (Categorical)
12. Displaced - Whether the student is a displaced person (Categorical)
13. Educational special needs - Whether the student has any special educational needs (Categorical)
14. Debtor - Whether the student is a debtor (Categorical)
15. Tuition fees up to date	- Whether the student's tuition fees are up to date (Categorical)
16. Gender - The gender of the student (Categorical)
17. Scholarship holder - Whether the student is a scholarship holder (Categorical)
18. Age at enrollment - The age of the student at the time of enrollment (Numerical)
19. International - Whether the student is an international student (Categorical)
20. Curricular units 1st sem (credited)	- The number of curricular units credited by the student in the first semester (Numerical)
21. Curricular units 1st sem (enrolled)	- The number of curricular units enrolled by the student in the first semester (Numerical)
22. Curricular units 1st sem (evaluations) - The number of curricular units evaluated by the student in the first semester (Numerical)
23. Curricular units 1st sem (approved) - The number of curricular units approved by the student in the first semester (Numerical)
24. Curricular units 1st sem (grade) - The number of curricular units grade by the student in the first semester (Numerical)
25. Curricular units 1st sem (without evaluations) - The number of curricular units without evaluations by the student in the first semester (Numerical)
26. Curricular units 2nd sem (credited) - The number of curricular units credited by the student in the second semester (Numerical)
27. Curricular units 2nd sem (enrolled) - The number of curricular units enrolled by the student in the second semester (Numerical)
28. Curricular units 2nd sem (evaluations) - The number of curricular units evaluated by the student in the second semester (Numerical)
29. Curricular units 2nd sem (approved)	- The number of curricular units approved by the student in the second semester (Numerical)
30. Curricular units 2nd sem (grade) - The number of curricular units grade by the student in the second semester (Numerical)
31. Curricular units 2nd sem (without evaluations) - The number of curricular units without evaluations by the student in the second semester (Numerical)
32. Unemployment rate - The unemployment rate (Numerical)
33. Inflation rate - The inflation rate (Numerical)
34. GDP - The GDP	(Numerical)
35. Target	Graduated/Dropout/Enrolled	(Categorical)


## Project Overview
The target variable in this dataset originally had three unique values: "Dropout," "Graduate," and "Enrolled." 
![Target_3](https://github.com/sylwiaSekula/Student_Graduation/assets/110921660/275b878a-46a9-4f65-adee-315572a88e35) 

However, I decided to exclude "Enrolled" from the target column since it represents students who are still studying.
![Target_2](https://github.com/sylwiaSekula/Student_Graduation/assets/110921660/6b88d357-aece-436f-a224-b546418d4dd6)

    Graduate    2209
    Dropout     1421

After I excluded "Enrolled" from the target column the dataset contains 3630 entries. 

The target data was encoded using Label Encoder to convert the categorical labels into numerical values suitable for machine learning algorithms. I scaled the numerical columns in the dataset using StandardScaler.
I performed feature selection for each model using the Sequential Feature Selector from the scikit-learn library, selecting 20 relevant features using forward selection.
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

     Dropout       0.91      0.84      0.87       467
    Graduate       0.90      0.95      0.92       731

    accuracy                           0.90      1198
    macro avg       0.91      0.89     0.90      1198
    weighted avg    0.91      0.90     0.90      1198
    
 ![logistic regression_confusion_matrix](https://github.com/sylwiaSekula/Student_Graduation/assets/110921660/58b939b6-fc02-40e6-8bcd-d8b770d11898)

    
Support Vector Classifier 
              
                precision    recall  f1-score   support
     Dropout       0.91      0.84      0.87       467
    Graduate       0.90      0.95      0.92       731

    accuracy                           0.90      1198
    macro avg      0.91      0.89      0.90      1198
    weighted avg   0.90      0.90      0.90      1198
    
![Support Vector Classifier_confusion_matrix](https://github.com/sylwiaSekula/Student_Graduation/assets/110921660/03fe8149-8c0d-436f-bce8-9c150915eea7)

Light Gradient Boosting Machine
              
                precision    recall  f1-score   support
     Dropout       0.93      0.84      0.88       467
    Graduate       0.90      0.96      0.93       731

    accuracy                           0.91      1198
    macro avg       0.92      0.90     0.91      1198
    weighted avg    0.91      0.91     0.91      1198
    
![Light Gradient Boosting Machine_confusion_matrix](https://github.com/sylwiaSekula/Student_Graduation/assets/110921660/6f233e7e-3ec0-4297-99e1-e67b32deb80a)
    

In the confusion matrixes, you can see the counts of true negatives (correctly predicted as "Dropout"), false positives (actually "Dropout" but predicted as "Graduate"), false negatives (actually "Graduate" but predicted as "Dropout"), and true positives (correctly predicted as "Graduate"). LGBM has the highest number of true positives (701), indicating that it correctly predicts "Graduate" students. LGBM has also the lowest number of true negatives (391), signifying that it makes fewer incorrect predictions for "Dropout" students.

## Conclusions:
The results suggest that the machine learning models applied in this project are well-suited for the task of predicting student graduation status. All three models demonstrated good overall performance with high precision, recall, F1-scores, and accuracy .LGBM has the highest accuracy - 91%. SVC and Logistic Regression are slightly lower - at 90%. LGBM shows the best overall performance with higher precision, recall, and F1-scores for both classes compared to the other two models. 
