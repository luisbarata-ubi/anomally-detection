from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
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
import itertools

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
def evaluate_model(var_smoothing_value, X, y):
    """
    Trains and evaluates a Gaussian Naive Bayes model using K-Fold
    cross-validation. Returns average metrics across all folds.
    """
    gnb_model = GaussianNB(var_smoothing=var_smoothing_value)

    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    fold_metrics = []

    for train_index, test_index in kf.split(X):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Split training data into (train+val) and test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.3,
            random_state=48,
            stratify=y_train_full
        )
        X_val, X_test_fold, y_val, y_test_fold = train_test_split(
            X_temp,
            y_temp,
            test_size=0.33,
            random_state=48,
            stratify=y_temp
        )


        ############################################################
        ############################################################
        ############################################################

        # If desired, uncomment one of the sampling approaches below.
        # 1) SMOTE + ENN
        #smote_enn = SMOTEENN(random_state=48)
        #X_train, y_train = smote_enn.fit_resample(X_train, y_train)

        # 2) SMOTE (over-sampling)
        #smote = SMOTE(random_state=48)
        #X_train, y_train = smote.fit_resample(X_train, y_train)

        # 3) Random under-sampling
        #undersampler = RandomUnderSampler(random_state=48)
        #X_train, y_train = undersampler.fit_resample(X_train, y_train)

        ############################################################
        ############################################################
        ############################################################

        gnb_model.fit(X_train, y_train)
        y_pred = gnb_model.predict(X_test)

        # Calculate metrics for this fold
        fold_metrics.append(calculate_metrics(y_test, y_pred))

    # Calculate average metrics across all folds
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics if metric != "confusion_matrix"])
        for metric in fold_metrics[0] if metric != "confusion_matrix"
    }
    avg_metrics["confusion_matrix"] = fold_metrics[0]["confusion_matrix"]
    return avg_metrics

# Redirect print statements to a file
# Change the file name as needed

output_file_path = sys.argv[2] + "/gaussian_inbalanced_nohyper.txt"
#output_file_path = sys.argv[2] + "/gaussian_over_nohyper.txt"
#output_file_path = sys.argv[2] + "/gaussian_under_nohyper.txt"
#output_file_path = sys.argv[2] + "/gaussian_hybrid_nohyper.txt"
sys.stdout = open(output_file_path, 'w')

if __name__ == "__main__":
    # Load dataset and prepare data (you can replace this with your own data loading function)
    # X, y should be features and labels
    file_path = sys.argv[1]
    save_directory = sys.argv[2]

    print("**** Imbalaced ****\n")
    #print("**** Over ****\n")
    #print("**** Under ****\n")
    #print("**** Hybrid ****\n")
    print("ficheiro origem: ", file_path, "\n")
    print("pasta de resultados: ", save_directory, "\n")

    # Set your email details
    email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Imbalaced) - noHyper - {save_directory}")
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Over) - noHyper - {save_directory}")
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Under) - noHyper - {save_directory}")
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Hybrid) - noHyper - {save_directory}")
    email_body = "Your script has completed successfully."

    # Start time
    start_time = time.time()

    # Load data
    dataset = pd.read_csv(file_path)

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    total_rows = len(dataset)
    percentage_anomalies = (num_anomalies / total_rows) * 100
    print("Percentage of rows where anomaly = 1:", percentage_anomalies)

    # Ensure dataset has no NaNs and proper type conversions
    dataset = dataset.dropna()  # Remove missing values, if any
    dataset['anomaly'] = dataset['anomaly'].astype(int)

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    # Print dataset shape
    print(f"Dataset size: {dataset.shape} rows, {dataset.shape} columns")
    print(f"Dataset size after exclusion: {dataset.shape} rows, {dataset.shape} columns")

    # Separate features and target
    X = dataset.drop(['anomaly', 'trace_id', 'timestamp'], axis=1)
    y = dataset['anomaly']

    # Generate mappings (if you have categorical columns)
    mappings = {}
    for column in ['source', 'target', 'succ']:
        mapping = myPhDlibs.generate_mapping(X, column)
        print(f"{column.capitalize()} Mapping:", mapping)
        mappings[column] = mapping

    # Apply mappings if needed
    for column, mapping in mappings.items():
        unique_values = X[column].unique()
        print(f"Unique values in {column}: {unique_values}")
        if all(value in mapping for value in unique_values):
            X[column] = X[column].map(mapping)

    # ------------------------------------------------------
    # Here we define a fixed var_smoothing for GaussianNB.
    # You can adjust as needed (e.g., 1e-9, 1e-8, etc.)
    # ------------------------------------------------------
    var_smoothing_fixed = 1e-9

    # Evaluate the model using a *fixed* var_smoothing
    fixed_metrics = evaluate_model(var_smoothing_fixed, X, y)

    # Print the results
    print(f"Fixed var_smoothing: {var_smoothing_fixed}")
    print(f"Performance Metrics: {fixed_metrics}")

    # End time
    end_time = time.time()
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

    # Optionally save performance data
    # myPhDlibs.write_performance_data(start_time, end_time, execution_time, memory_used, cpu_used, save_path=save_directory)

    # Send the email
    myPhDlibs.send_email(email_subject, email_body)
    print("Email sent successfully.")

    # Final memory cleanup
    memory_cleanup()
    print("Script completed successfully.")