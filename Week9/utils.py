import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def load_prepare_data(path):
    
    data = pd.read_csv(path)
    
    print('Show dataset:')
    print(data.head().T)
    print(50*'-')
    
    print('Informations:')
    print(data.info())
    print(50*'-')
    
    print('Column names:')
    print(data.columns)
    print(50*'-')
    
    print('Statistical reports:')
    print(data.describe().T)
    
    return data


def load_classifier(X, y):
    clf = input('Please enter your classifier, decision tree(dt)/knn/svm/random forest(rf)/adaboost(ada): ').lower()

    if clf == ('dt' or 'decision tree'):
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        max_depth = input('Please enter the maximum depth of the decision tree (default is None): ')
        if max_depth:
            max_depth = int(max_depth)
        else:
            max_depth = None
        criterion = input('Please enter the criterion for the decision tree (default is gini): ') or 'gini'
        print(50*'-')
        classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        
    elif clf == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        n_neighbors = int(input('Please enter the number of neighbors for KNN (default is 5): ') or 5)
        weights = input('Please enter the type of weight function for KNN (default is uniform): ') or 'uniform'
        print(50*'-')
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    elif clf == 'svm':
        from sklearn.svm import SVC
        kernel = input('Please enter the type of kernel for SVM (default is rbf): ') or 'rbf'
        C = float(input('Please enter the regularization parameter for SVM (default is 1.0): ') or 1.0)
        print(50*'-')
        classifier = SVC(kernel=kernel, C=C)

    elif clf == ('rf' or 'random forest'):
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = int(input('Please enter the number of trees for random forest (default is 100): ') or 100)
        max_depth = input('Please enter the maximum depth of each tree for random forest (default is None): ') 
        if max_depth:
            max_depth = int(max_depth)
        else:
            max_depth = None
        criterion = input('Please enter the criterion for random forest (default is gini): ') or 'gini'
        print(50*'-')
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

    elif clf == ('ada' or 'adaboost'):
        from sklearn.ensemble import AdaBoostClassifier
        n_estimators = int(input('Please enter the number of estimators for AdaBoost (default is 50): ') or 50)
        learning_rate = float(input('Please enter the learning rate for AdaBoost (default is 1.0): ') or 1.0)
        print(50*'-')
        classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    else:
        print('Your classifier was not found!')
        classifier = None

    if classifier is not None:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the classifier on the training data
        classifier.fit(X_train, y_train)

        # Predict the labels of the test data
        y_pred = classifier.predict(X_test)

        # Print the classification report
        print('Reports:')
        print(classification_report(y_test, y_pred))
        print(50*'-')
        
        # Print the confusion matrix
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print(50*'-')
        
        if clf == 'dt':
            fig, ax = plt.subplots(figsize=(50, 40))
            plot_tree(classifier, ax=ax, filled=True, fontsize=12)
            plt.show()
        print('Classifier is loaded successfully.')

    else:
        print('Classifier was not loaded!')

    return classifier


def perform_grid_search(X, y, clf, params, scoring_list, cross_validation):
    """
    Performs grid search on the classifier using the given parameters.

    Parameters:
    X (array-like): The feature matrix.
    y (array-like): The target vector.
    clf (estimator): The classifier object.
    params (dict): The dictionary of hyperparameters to search over.
    scoring_list: The list of scoring functions (f1, recall, accuracy, ...)
    cross_validation(int): The integer number for spliting dataset.
    Returns:
    GridSearchCV: The grid search object.
    """

    for i in scoring_list:
        print(f'Scoring {i}: \n')
        
        gsh = GridSearchCV(clf, param_grid=params, scoring= i, 
                           cv=cross_validation, n_jobs=-1, verbose=1)
        gsh.fit(X, y)

        print(f'The best Estimators: {gsh.best_estimator_}')
        print(30*'-')
        print(f'The best Score: {gsh.best_score_}')
        print(30*'-')
        print(f'The best Parameters: {gsh.best_params_}')
        print(50*'#')
    return gsh