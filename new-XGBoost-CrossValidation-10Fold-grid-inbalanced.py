from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold, GridSearchCV
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


# Define the search space for hyperparameters
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# Redirect print statements to a file
#output_file_path = sys.argv[2] + "/xgboost_inbalanced_grid.txt"
#output_file_path = sys.argv[2] + "/xgboost_over_grid.txt"
#output_file_path = sys.argv[2] + "/xgboost_under_grid.txt"
output_file_path = sys.argv[2] + "/xgboost_hybrid_grid.txt"
sys.stdout = open(output_file_path, 'w')

if __name__ == "__main__":
    # Load dataset and prepare data (you can replace this with your own data loading function)
    # X, y should be features and labels
    file_path = sys.argv[1]
    save_directory = sys.argv[2]

    #print("**** Imbalaced ****\n")
    #print("**** Over ****\n")
    #print("**** Under ****\n")
    print("**** Hybrid ****\n")
    print("ficheiro origem: ", file_path, "\n")
    print("pasta de resultados: ", save_directory, "\n")

    # Set your email details
    #email_subject = (f"Script Completion Notification - XGBoost 10-Fold Cross Validation (Imbalaced) - Grid - {save_directory}")
    #email_subject = (f"Script Completion Notification - XGBoost 10-Fold Cross Validation (Over) - Grid - {save_directory}")
    #email_subject = (f"Script Completion Notification - XGBoost 10-Fold Cross Validation (Under) - Grid - {save_directory}")
    email_subject = (f"Script Completion Notification - XGBoost 10-Fold Cross Validation (Hybrid) - Grid - {save_directory}")
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
    #dataset.loc[dataset['latency'] < Q3, 'anomaly'] = 0

    num_anomalies = (dataset['anomaly'] == 1).sum()
    print("Number of rows where anomaly = 1:", num_anomalies)

    # Print the size of the dataset
    print(f"Dataset size: {dataset.shape} rows, {dataset.shape} columns")

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

    print(X)


    ############################################################
    ############################################################
    ############################################################

    # Apply hybrid sampling (SMOTE + ENN) to the entire dataset
    #smote_enn = SMOTEENN(random_state=48)
    #X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    # Apply oversampling (SMOTE) to the entire dataset
    #smote = SMOTE(random_state=48)
    #X_resampled, y_resampled = smote.fit_resample(X, y)

    # Apply undersampling to the entire dataset
    #undersampler = RandomUnderSampler(random_state=48)
    #X_resampled, y_resampled = undersampler.fit_resample(X, y)

    X_resampled = X
    y_resampled = y

    ############################################################
    ############################################################
    ############################################################

    # Define model
    xgb_model = XGBClassifier(
        random_state=48,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1',
        cv=10,  # 10-fold cross-validation
        verbose=1,
        n_jobs=-1
    )

    # Perform hyperparameter tuning
    print("Starting GridSearchCV...")
    grid_search.fit(X_resampled, y_resampled)

    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Train model with best parameters on the entire dataset
    best_model = grid_search.best_estimator_

    # Metrics calculation on resampled data
    y_pred_resampled = best_model.predict(X_resampled)
    accuracy, precision, recall, f1, auc = calculate_metrics(y_resampled, y_pred_resampled)

    print("Metrics on Resampled Dataset:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")
    
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
