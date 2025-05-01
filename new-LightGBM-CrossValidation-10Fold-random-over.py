from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import gc  # Garbage collection
import sys
import time
import myPhDlibs
import random

# Function to clean up memory
def memory_cleanup():
    gc.collect()

# Define the function for training and calculating metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "f1": f1_score(y_true, y_pred, zero_division=1),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # For interpretability
    }
    return metrics

# Define the function for model evaluation
def evaluate_model(params, X, y):
    lgbm_model = LGBMClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        max_depth=params['max_depth'],
        random_state=48
    )

    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    fold_metrics = []

    for train_index, test_index in kf.split(X):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_train_full, y_train_full, test_size=0.3, random_state=48, stratify=y_train_full
        )
        X_val, X_test_fold, y_val, y_test_fold = train_test_split(
            X_temp, y_temp, test_size=0.33, random_state=48, stratify=y_temp
        )

        ############################################################
        ############################################################
        ############################################################

        # Apply hybrid sampling (SMOTE + ENN) to the training data
        #smote_enn = SMOTEENN(random_state=48)
        #X_train, y_train = smote_enn.fit_resample(X_train, y_train)

        # Apply oversampling (SMOTE) to the training data
        smote = SMOTE(random_state=48)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Apply undersampling to the training data
        #undersampler = RandomUnderSampler(random_state=48)
        #X_train, y_train = undersampler.fit_resample(X_train, y_train)

        ############################################################
        ############################################################
        ############################################################
        lgbm_model.fit(X_train, y_train)
        y_pred = lgbm_model.predict(X_test)

        # Calculate metrics for the fold
        fold_metrics.append(calculate_metrics(y_test, y_pred))

    # Calculate average metrics across all folds
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics if metric != "confusion_matrix"])
        for metric in fold_metrics[0] if metric != "confusion_matrix"
    }
    avg_metrics["confusion_matrix"] = fold_metrics[0]["confusion_matrix"]  # Return one example matrix for clarity
    return avg_metrics


# Random Search Implementation
def random_search(param_space, n_iter, X, y):
    best_params = None
    best_score = -float('inf')
    best_metrics = {}

    for i in range(n_iter):
        params = {
            'n_estimators': random.choice(param_space['n_estimators']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'num_leaves': random.choice(param_space['num_leaves']),
            'max_depth': random.choice(param_space['max_depth']),
        }
        metrics = evaluate_model(params, X, y)
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_params = params
            best_metrics = metrics
        print(f"Iteration {i + 1}/{n_iter}, Params: {params}, F1 Score: {metrics['f1']:.4f}")

    return best_params, best_metrics



# Redirect print statements to a file
#output_file_path = sys.argv[2] + "/lightgbm_inbalanced_random.txt"
output_file_path = sys.argv[2] + "/lightgbm_over_random.txt"
#output_file_path = sys.argv[2] + "/lightgbm_under_random.txt"
#output_file_path = sys.argv[2] + "/lightgbm_hybrid_random.txt"
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
    #email_subject = (f"Script Completion Notification - LightGBM 10-Fold Cross Validation (Imbalaced) - Random - {save_directory}")
    email_subject = (f"Script Completion Notification - LightGBM 10-Fold Cross Validation (Over) - Random - {save_directory}")
    #email_subject = (f"Script Completion Notification - LightGBM 10-Fold Cross Validation (Under) - Random - {save_directory}")
    #email_subject = (f"Script Completion Notification - LightGBM 10-Fold Cross Validation (Hybrid) - Random - {save_directory}")
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
    #QValue = 1 - (percentage_anomalies / 100)

    # Calculate quartiles
    #Q3 = dataset['latency'].quantile(QValue)
    #print("Q value:", QValue)
    #print("Q latency value:", Q3)

    # Update is_anomaly column
    #dataset.loc[dataset['latency'] < Q3, 'anomaly'] = 0

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    # Print the size of the dataset
    print(f"Dataset size: {dataset.shape} rows, {dataset.shape} columns")

    ############################################################
    ############################################################
    ############################################################
    # Exclude rows with latency less than 10ms (apenas para o meu algoritmo)
    #dataset = dataset[dataset['latency'] >= 10]
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

            
    param_space = {
        'n_estimators': range(50, 301, 10),
        'learning_rate': [0.01, 0.03, 0.1, 0.2, 0.3],
        'num_leaves': range(20, 151),
        'max_depth': range(3, 16),
    }

    n_iter = 50  # Number of random search iterations
    best_params, best_metrics = random_search(param_space, n_iter, X, y)

    print(f"Best Parameters: {best_params}")
    print(f"Performance Metrics: {best_metrics}")

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
