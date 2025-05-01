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

# ---- Hyperopt imports ----
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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
    """
    Trains and evaluates a Gaussian Naive Bayes model using K-Fold 
    cross-validation. Returns average metrics across all folds.
    """
    gnb_model = GaussianNB(var_smoothing=params['var_smoothing'])

    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    fold_metrics = []

    for train_index, test_index in kf.split(X):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Split training data into (train+val) and test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_train_full, y_train_full,
            test_size=0.3,
            random_state=48,
            stratify=y_train_full
        )
        X_val, X_test_fold, y_val, y_test_fold = train_test_split(
            X_temp, y_temp,
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
        undersampler = RandomUnderSampler(random_state=48)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

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

# ---- Hyperopt Objective Function ----
def objective(params):
    """
    Objective function for Hyperopt.
    We aim to MAXIMIZE F1, so we MINIMIZE -F1.
    """
    metrics = evaluate_model(params, X, y)
    loss = -metrics['f1']  # negative F1 for maximizing F1
    return {
        'loss': loss,
        'status': STATUS_OK,
        'metrics': metrics  # Store full metrics for retrieval
    }



# Redirect print statements to a file
# Change the file name as needed

#output_file_path = sys.argv[2] + "/gaussian_inbalanced_hyperopt.txt"
#output_file_path = sys.argv[2] + "/gaussian_over_hyperopt.txt"
output_file_path = sys.argv[2] + "/gaussian_under_hyperopt.txt"
#output_file_path = sys.argv[2] + "/gaussian_hybrid_hyperopt.txt"
sys.stdout = open(output_file_path, 'w')

if __name__ == "__main__":
    # Load dataset and prepare data (you can replace this with your own data loading function)
    # X, y should be features and labels
    file_path = sys.argv[1]
    save_directory = sys.argv[2]

    #print("**** Imbalaced ****\n")
    #print("**** Over ****\n")
    print("**** Under ****\n")
    #print("**** Hybrid ****\n")
    print("ficheiro origem: ", file_path, "\n")
    print("pasta de resultados: ", save_directory, "\n")

    # Set your email details
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Imbalaced) - HyperOpt - {save_directory}")
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Over) - HyperOpt - {save_directory}")
    email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Under) - HyperOpt - {save_directory}")
    #email_subject = (f"Script Completion Notification - GaussianNB 10-Fold Cross Validation (Hybrid) - HyperOpt - {save_directory}")
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

    # Prepare features and target
    X = dataset.drop(['anomaly', 'trace_id', 'timestamp'], axis=1)
    y = dataset['anomaly']

    # Generate mappings (if you need to transform categorical columns)
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

    # Make X and y globally accessible to the objective function
    global X, y

    # ---- Define the Hyperopt search space ----
    # Option 1: Discrete choices:
    # space = {
    #    'var_smoothing': hp.choice('var_smoothing', [1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    # }

    # Option 2: A range for var_smoothing using loguniform (common for NB):
    space = {
        # This will sample var_smoothing from ~ log(1e-9) to log(1e-5)
        # i.e., range ~ [-20.72, -11.51], but you can adjust as needed
        'var_smoothing': hp.loguniform('var_smoothing', -20, -11)
    }

    # Keep references if using hp.choice, so you can map them back. (Not needed if using loguniform.)

    # Hyperopt trials object
    trials = Trials()

    # Number of evaluations (trials) for Hyperopt
    max_evals = 30

    print(f"Starting hyperopt with max_evals={max_evals}...\n")

    # Run Hyperopt fmin
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,   # Tree-structured Parzen Estimator
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(48)  # for reproducibility
    )

    # If you used hp.choice for var_smoothing, you'd map back from indices, e.g.:
    # var_smoothing_options = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    # best_var_smoothing = var_smoothing_options[best['var_smoothing']]

    # If using hp.loguniform, best['var_smoothing'] is already the numeric value
    best_var_smoothing = best['var_smoothing']

    best_params = {
        'var_smoothing': best_var_smoothing
    }

    # Find the best trial
    best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
    best_metrics = best_trial['result']['metrics']

    print(f"Best Parameters (raw): {best}")
    print(f"Best Parameters (mapped): {best_params}")
    print(f"Performance Metrics: {best_metrics}")

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

    # Optionally, save performance data
    # myPhDlibs.write_performance_data(start_time, end_time, execution_time, memory_used, cpu_used, save_path=save_directory)

    # Send the email
    myPhDlibs.send_email(email_subject, email_body)
    print("Email sent successfully.")

    # Final memory cleanup
    memory_cleanup()
    print("Script completed successfully.")