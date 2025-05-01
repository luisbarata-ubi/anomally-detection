from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import gc  # Garbage collection
import sys
import time
import myPhDlibs

# Function to clean up memory
def memory_cleanup():
    gc.collect()

# Define function for calculating metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, auc

# Redirect print statements to a file
output_file_path = sys.argv[2] + "/my_inbalanced_nohyper.txt"
# output_file_path = sys.argv[2] + "/my_over_nohyper.txt"
# output_file_path = sys.argv[2] + "/my_under_nohyper.txt"
# output_file_path = sys.argv[2] + "/my_hybrid_nohyper.txt"
sys.stdout = open(output_file_path, 'w')

if __name__ == "__main__":
    # Load dataset and prepare data
    file_path = sys.argv[1]
    save_directory = sys.argv[2]

    print("**** Imbalaced ****\n")
    #print("**** Oversampling ****\n")
    #print("**** Undersampling ****\n")
    #print("**** Hybridsampling ****\n")
    print("ficheiro origem: ", file_path, "\n")
    print("pasta de resultados: ", save_directory, "\n")

    # Set email details
    email_subject = f"Script Completion Notification - My 10-Fold Cross Validation (Imbalaced) - NoHyper - {save_directory}"
    #email_subject = f"Script Completion Notification - My 10-Fold Cross Validation (Oversampling) - NoHyper - {save_directory}"
    #email_subject = f"Script Completion Notification - My 10-Fold Cross Validation (Undersampling) - NoHyper - {save_directory}"
    #email_subject = f"Script Completion Notification - My 10-Fold Cross Validation (Hybridsampling) - NoHyper - {save_directory}"
    email_body = "Your script has completed successfully."

    # Start time
    start_time = time.time()

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

    ############################################################
    ############################################################
    ############################################################
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
            
    # Define XGBoost model with fixed hyperparameters
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=48,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # KFold cross-validation with 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores = [], [], [], [], []

    for fold_number, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Split data into 70% train, 20% validation, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=48, stratify=y_train_full)
        X_val, X_test_fold, y_val, y_test_fold = train_test_split(X_temp, y_temp, test_size=0.33, random_state=48, stratify=y_temp)

        ############################################################
        ############################################################
        ############################################################

        # Apply hybrid sampling (SMOTE + ENN) to the training data
        #smote_enn = SMOTEENN(random_state=48)
        #X_train, y_train = smote_enn.fit_resample(X_train, y_train)

        # Apply oversampling (SMOTE) to the training data
        #smote = SMOTE(random_state=48)
        #X_train, y_train = smote.fit_resample(X_train, y_train)

        # Apply undersampling to the training data
        #undersampler = RandomUnderSampler(random_state=48)
        #X_train, y_train = undersampler.fit_resample(X_train, y_train)

        ############################################################
        ############################################################
        ############################################################

        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        # Initialize XGBoost model
        xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, early_stopping_rounds = 3)

        # Train the model
        training_start_time = time.time()
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        training_time = time.time() - training_start_time

        # Validate the model
        y_val_pred = xgb_model.predict(X_val)
        accuracy_val, precision_val, recall_val, f1_val, auc_val = calculate_metrics(y_val, y_val_pred)

        # Test the model
        y_test_pred = xgb_model.predict(X_test_fold)
        accuracy_test = accuracy_score(y_test_fold, y_test_pred)

        # Print and store metrics for each fold
        print(f"Fold {fold_number} - Validation Accuracy: {accuracy_val}, Precision: {precision_val}, Recall: {recall_val}, F1: {f1_val}, AUC: {auc_val}")
        accuracy_scores.append(accuracy_val)
        precision_scores.append(precision_val)
        recall_scores.append(recall_val)
        f1_scores.append(f1_val)
        auc_scores.append(auc_val)

    # Mean of the metrics across all folds
    print(f"Mean Accuracy: {np.mean(accuracy_scores)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")
    print(f"Mean F1 Score: {np.mean(f1_scores)}")
    print(f"Mean AUC: {np.mean(auc_scores)}")

    # End time and execution time
    end_time = time.time()
    execution_time = end_time - start_time
    memory_used = myPhDlibs.get_memory_usage()
    cpu_used = myPhDlibs.get_cpu_usage()

    print(f"Execution time: {execution_time} seconds")
    print(f"Memory used: {memory_used} bytes")
    print(f"CPU used: {cpu_used}%")

    # Send the email
    myPhDlibs.send_email(email_subject, email_body)
    print("Email sent successfully.")

    # Final memory cleanup
    memory_cleanup()
    print("Script completed successfully.")
