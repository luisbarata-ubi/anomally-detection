from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
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

# Define the objective function for hyperparameter optimization
def objective(params):
    rf_model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        max_features=params['max_features'],
        min_samples_split=int(params['min_samples_split']),
        random_state=48,
        n_jobs=-1
    )

    # Split the dataset into training, validation, and test sets (70% train, 20% validation, 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=48, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=48, stratify=y_temp)  # 33% of 30% = 10%

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

    # Train the model
    training_start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - training_start_time

    # Print training and testing times
    print(f"Training Time: {training_time} seconds")

    # Make predictions on the validation and test sets
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    # Calculate train and test accuracies
    train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Calculate additional metrics
    precision = precision_score(y_val, y_val_pred, zero_division=1)
    recall = recall_score(y_val, y_val_pred, zero_division=1)
    f1 = f1_score(y_val, y_val_pred, zero_division=1)
    auc = roc_auc_score(y_val, y_val_pred)
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    # Cross-validation for mean accuracy and standard deviation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=48)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Mean Accuracy: {mean_accuracy}, Accuracy Std Dev: {std_accuracy}")

    # Return negative F1 score for optimization (Hyperopt minimizes the function)
    return {'loss': -f1, 'status': STATUS_OK, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}

# Define the search space for hyperparameters
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'max_features': hp.uniform('max_features', 0.1, 1.0),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# Redirect print statements to a file
#output_file_path = sys.argv[2] + "/randomforest_inbalanced_hyperopt.txt"
output_file_path = sys.argv[2] + "/randomforest_over_hyperopt.txt"
#output_file_path = sys.argv[2] + "/randomforest_under_hyperopt.txt"
#output_file_path = sys.argv[2] + "/randomforest_hybrid_hyperopt.txt"
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
    #email_subject = (f"Script Completion Notification - Random Forest 10-Fold Cross Validation (Imbalaced) - HyperOPT - {save_directory}")
    email_subject = (f"Script Completion Notification - Random Forest 10-Fold Cross Validation (Over) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - Random Forest 10-Fold Cross Validation (Under) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - Random Forest 10-Fold Cross Validation (Hybrid) - HyperOPT - {save_directory}")
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

    print(X)

    # Hyperparameter optimization using Hyperopt
    trials = Trials()
    best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,  # Number of evaluations
            trials=trials,
            rstate=np.random.default_rng(48)
        )

    # Convert parameters from float to int where necessary
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])

    print(f"Best Hyperparameters: {best_params}")

    ###############################################################

    # Initialize KFold cross-validator
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Lists to store metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    training_times = []
    testing_times = []

    for train_index, test_index in kf.split(X):
        # Split data into training and testing sets
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Further split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2222, random_state=42)  # 20/90 ~ 0.2222

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

        # Initialize the best model using the best hyperparameters
        best_model = RandomForestClassifier(
            n_estimators=int(best_params['n_estimators']),
            max_depth=int(best_params['max_depth']),
            min_samples_split=int(best_params['min_samples_split']),
            min_samples_leaf=int(best_params['min_samples_leaf']),
            max_features=best_params['max_features'],
            random_state=48,
            n_jobs=-1
        )

        # Fit the best model
        training_start_time = time.time()
        best_model.fit(X_train, y_train)
        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        # Make predictions on the test set
        testing_start_time = time.time()
        y_pred = best_model.predict(X_test)

        testing_end_time = time.time()
        testing_time = testing_end_time - testing_start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_test)
        precision = precision_score(y_test, y_test)
        recall = recall_score(y_test, y_test)
        f1 = f1_score(y_test, y_test)
        auc = roc_auc_score(y_test, y_test)

        # Append metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
        training_times.append(training_time)
        testing_times.append(testing_time)

    # Calculate and print average metrics
    print(f"Mean Accuracy: {np.mean(accuracy_scores)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")
    print(f"Mean F1 Score: {np.mean(f1_scores)}")
    print(f"Mean AUC: {np.mean(auc_scores)}")
    print(f"Mean Training Time: {np.mean(training_times)} seconds")
    print(f"Mean Testing Time: {np.mean(testing_times)} seconds")



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
