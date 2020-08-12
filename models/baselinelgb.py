from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
import numpy as np
import pandas as pd


def model(training_data, testing_data, n_folds=5):
    """Train and test a light gradient boosting model using
    cross validation.

    Parameters
    --------
        training_data (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        testing_data (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
        n_folds (int, default = 5): number of folds to use for cross validation

    Return
    --------
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
    """

    # Extract the labels for training
    y_train = training_data['TARGET']

    # Remove the feature_labels and target
    X_train = training_data.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_ids = testing_data['SK_ID_CURR']
    X_test = testing_data.drop(columns=['SK_ID_CURR'])

    # One hot encode features
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align the dataframes by the columns
    X_train, X_test = X_train.align(X_test, join='inner', axis=1)

    print('Training Data Shape: ', X_train.shape)
    print('Testing Data Shape: ', X_test.shape)

    # Extract feature names
    feature_names = list(X_train.columns)

    # Convert to np arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(X_test.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(X_train.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(X_train):
        # Training data for the fold
        train_features, train_labels = X_train[train_indices], y_train[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = X_train[valid_indices], y_train[valid_indices]

        # Create the model
        the_model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                       class_weight='balanced', learning_rate=0.05,
                                       reg_alpha=0.1, reg_lambda=0.1,
                                       subsample=0.8, n_jobs=-1, random_state=50)

        # Train the model
        the_model.fit(train_features, train_labels, eval_metric='auc',
                      eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                      eval_names=['valid', 'train'], categorical_feature='auto',
                      early_stopping_rounds=100, verbose=200)

        # Record the best iteration
        best_iteration = the_model.best_iteration_

        # Record the feature importances
        feature_importance_values += the_model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += the_model.predict_proba(X_test, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = the_model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

        # Record the best score
        valid_score = the_model.best_score_['valid']['auc']
        train_score = the_model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del the_model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(y_train, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics
