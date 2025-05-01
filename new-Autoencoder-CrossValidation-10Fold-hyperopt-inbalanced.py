from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
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

# Build the Autoencoder model
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    
    # Compile the model
    autoencoder.compile(optimizer=Adam(), loss='mse')
    
    return autoencoder

# Define the function for calculating reconstruction error and metrics
def calculate_metrics(y_true, y_pred, threshold):
    # Compute the reconstruction error
    reconstruction_error = np.mean(np.power(y_true - y_pred, 2), axis=1)
    
    # Predict anomalies based on the threshold
    y_pred_anomaly = np.where(reconstruction_error > threshold, 1, 0)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_anomaly)
    precision = precision_score(y_true, y_pred_anomaly, zero_division=1)
    recall = recall_score(y_true, y_pred_anomaly, zero_division=1)
    f1 = f1_score(y_true, y_pred_anomaly, zero_division=1)
    auc = roc_auc_score(y_true, y_pred_anomaly)
    
    return accuracy, precision, recall, f1, auc

# Define the objective function for hyperparameter optimization
def objective(params):
    # KFold cross-validation (with 10 folds)
    kf = KFold(n_splits=10, shuffle=True, random_state=48)

    # Lists to store metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    
    fold_number = 1
    for train_index, test_index in kf.split(X):
        # Split the data into train (90%) and test (10%) for this fold
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        # Further split training data into 70% train and 30% temp (for validation)
        X_train, X_temp, y_train, y_temp = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=48, stratify=y_train_full)

        # Split temp set into 20% validation and 10% test
        X_val, X_test_fold, y_val, y_test_fold = train_test_split(X_temp, y_temp, test_size=0.33, random_state=48, stratify=y_temp)  # 33% of 30%

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

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_fold)

        # Build and train the Autoencoder
        training_start_time = time.time()
        autoencoder = build_autoencoder(X_train_scaled.shape[1])
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=32, validation_data=(X_val_scaled, X_val_scaled), verbose=0)
        training_time = time.time() - training_start_time

        # Print training and testing times
        print(f"Training Time: {training_time} seconds")

        # Predict reconstruction on validation data
        X_val_pred = autoencoder.predict(X_val_scaled)
        
        # Calculate reconstruction error for validation data and choose a threshold
        val_reconstruction_error = np.mean(np.power(X_val_scaled - X_val_pred, 2), axis=1)
        threshold = np.percentile(val_reconstruction_error, 95)  # Set threshold at the 95th percentile

        # Predict reconstruction on the test data
        X_test_pred = autoencoder.predict(X_test_scaled)

        # Calculate train and test accuracies
        train_accuracy = accuracy_score(y_train, autoencoder.predict(X_train))
        test_accuracy = accuracy_score(y_test_fold, X_test_pred)

        # Calculate metrics on the test set
        accuracy, precision, recall, f1, auc = calculate_metrics(y_test_fold, X_test_scaled, threshold)

        # Print and store metrics for each fold
        print(f"Fold {fold_number} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
        print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

        fold_number += 1

    
    # Calculate the mean of the metrics across all folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores)

    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    print(f"Mean AUC: {mean_auc}")

    # Return negative mean F1 score for optimization (since Hyperopt minimizes the loss)
    return {'loss': -mean_f1, 'status': STATUS_OK}


# Define the search space for hyperparameters
space = {
    'latent_dim': hp.choice('latent_dim', [4, 8, 16, 32]),
    'learning_rate': hp.loguniform('learning_rate', -5, -2),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'epochs': hp.choice('epochs', [50, 100, 200]),
    'optimizer': hp.choice('optimizer', ['adam', 'sgd']),
    'l2_regularization': hp.loguniform('l2_regularization', -6, -2)
}

# Redirect print statements to a file
output_file_path = sys.argv[2] + "/autoencoder_inbalanced_hyperopt.txt"
#output_file_path = sys.argv[2] + "/autoencoder_over_hyperopt.txt"
#output_file_path = sys.argv[2] + "/autoencoder_under_hyperopt.txt"
#output_file_path = sys.argv[2] + "/autoencoder_hybrid_hyperopt.txt"
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
    email_subject = (f"Script Completion Notification - Autoencoder 10-Fold Cross Validation (Imbalaced) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - Autoencoder 10-Fold Cross Validation (Over) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - Autoencoder 10-Fold Cross Validation (Under) - HyperOPT - {save_directory}")
    #email_subject = (f"Script Completion Notification - Autoencoder 10-Fold Cross Validation (Hybrid) - HyperOPT - {save_directory}")
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
