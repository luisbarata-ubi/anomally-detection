from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import gc  # Garbage collection
import sys
import time
import myPhDlibs
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Function to clean up memory
def memory_cleanup():
    gc.collect()

# Define the function for training and calculating metrics
def calculate_metrics(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, auc

# Define the objective function for hyperparameter optimization
def objective(params):
    # Create an XGBoost model with the given hyperparameters
    xgb_model = XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=48,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []

    for train_index, test_index in kf.split(X_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

        # Train the model
        xgb_model.fit(X_train, y_train)

        # Predict on the test set
        y_test_pred = xgb_model.predict(X_test)
        test_accuracy, precision, recall, f1, auc = calculate_metrics(y_test, y_test_pred)

        # Store metrics
        accuracy_scores.append(test_accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # Compute mean metrics across folds
    mean_f1 = np.mean(f1_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_auc = np.mean(auc_scores)

    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    print(f"Mean AUC: {mean_auc}")

    # Return the mean F1 score as the objective to maximize
    return {'loss': -mean_f1, 'status': STATUS_OK}

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}

# Redirect print statements to a file
#output_file_path = sys.argv[2] + "/my_inbalanced_hyperopt.txt"
output_file_path = sys.argv[2] + "/my_over_hyperopt.txt"
#output_file_path = sys.argv[2] + "/my_under_hyperopt.txt"
#output_file_path = sys.argv[2] + "/my_hybrid_hyperopt.txt"
sys.stdout = open(output_file_path, 'w')

if __name__ == "__main__":
    # Load dataset and prepare data (you can replace this with your own data loading function)
    # X, y should be features and labels
    file_path = sys.argv[1]
    save_directory = sys.argv[2]

    #print("**** Imbalaced ****\n")
    print("**** Over ****\n")
    #print("**** Under ****\n")
    #print("**** Hybrid ****\n")
    print("ficheiro origem: ", file_path, "\n")
    print("pasta de resultados: ", save_directory, "\n")

    # Set your email details
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Imbalaced) - HyperOPT - {save_directory}")
    email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Over) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Under) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Hybrid) - HyperOPT - {save_directory}")
    email_body = "Your script has completed successfully."

    # Start time
    start_time = time.time()

    # Specify the column containing latency information
    latency_column = 'latency'

    # Load data
    dataset = pd.read_csv(file_path)

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    total_rows = len(dataset)
    num_anomalies = (dataset['anomaly'] == 1).sum()
    percentage_anomalies = (num_anomalies / total_rows) * 100
    print("Percentage of rows where anomaly = 1:", percentage_anomalies)

    # Ensure dataset has no NaNs and proper type conversions
    dataset = dataset.dropna()  # Remove missing values, if any
    dataset['anomaly'] = dataset['anomaly'].astype(int)

    # Remove anomalies considered where latency less than Q3 value (apenas para o meu)
    QValue = 1 - (percentage_anomalies / 100)

    # Calculate quartiles
    Q3 = dataset['latency'].quantile(QValue)
    print("Q value:", QValue)
    print("Q latency value:", Q3)

    # Update is_anomaly column
    dataset.loc[dataset['latency'] < Q3, 'anomaly'] = 0

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    # Print the size of the dataset
    print(f"Dataset size: {dataset.shape} rows, {dataset.shape} columns")

    ############################################################
    ############################################################
    ############################################################
    # Exclude rows with latency less than 10ms (apenas para o meu algoritmo)
    dataset = dataset[dataset['latency'] >= 10]
    ############################################################
    ############################################################
    ############################################################

    # Print the size of the dataset after exclusion
    print(f"Dataset size after exclusion: {dataset.shape} rows, {dataset.shape} columns")

    X = dataset.drop(['anomaly', 'trace_id', 'timestamp'], axis=1)  # Features
    y = dataset['anomaly']  # Target variable

    # Generate mappings (optimized function)
    mappings = {}
    for column in ['source', 'target', 'succ']:
        mapping = myPhDlibs.generate_mapping(X, column)
        print(f"{column.capitalize()} Mapping:", mapping)
        mappings[column] = mapping

    # Apply mappings (optimized function)
    for column, mapping in mappings.items():
        unique_values = X[column].unique()
        print(f"Unique values in {column}: {unique_values}")
        if all(value in mapping for value in unique_values):
            X[column] = X[column].map(mapping)

    print(X)

    ############################################################
    ############################################################
    ############################################################

    # Apply hybrid sampling (SMOTE + ENN) to the training data
    #smote_enn = SMOTEENN(random_state=48)
    #X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    # Apply oversampling (SMOTE) to the training data
    smote = SMOTE(random_state=48)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Apply undersampling to the entire dataset before RandomizedSearchCV
    #undersampler = RandomUnderSampler(random_state=48)
    #X_resampled, y_resampled = undersampler.fit_resample(X, y)

    ############################################################
    ############################################################
    ############################################################

    # Hyperparameter optimization using Hyperopt
    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

    # Convert parameters from float to int where necessary
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])

    print(f"Best Hyperparameters: {best_params}")

    # End time
    end_time = time.time()

    # Execution time
    execution_time = end_time - start_time

    # Memory usage
    memory_used = myPhDlibs.get_memory_usage()

    # CPU usage
    cpu_used = myPhDlibs.get_cpu_usage()

    print("Script starting time: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("Script end time: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))))
    print("Execution time: {} seconds\n".format(execution_time))
    print("Memory used: {} bytes\n".format(memory_used))
    print("CPU used: {}%\n".format(cpu_used))

    #myPhDlibs.write_performance_data(start_time, end_time, execution_time, memory_used, cpu_used, save_path=save_directory)

    # Send the email
    myPhDlibs.send_email(email_subject, email_body)

    print("Email sent successfully.")


    # Final memory cleanup
    memory_cleanup()

    print("Script completed successfully.")
