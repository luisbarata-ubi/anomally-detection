from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import gc  # Garbage collection
import sys
import time
import myPhDlibs
from scipy.stats import randint, uniform 

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

# Define parameter distributions for random search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0)
}

# Redirect print statements to a file
output_file_path = sys.argv[2] + "/my_inbalanced_random.txt"
#output_file_path = sys.argv[2] + "/my_over_random.txt"
#output_file_path = sys.argv[2] + "/my_under_random.txt"
#output_file_path = sys.argv[2] + "/my_hybrid_random.txt"
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
    email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Imbalaced) - Random Search - {save_directory}")
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Over) - Random Search - {save_directory}")
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Under) - Random Search - {save_directory}")
    #email_subject = (f"Script Completion Notification - My 10-Fold Cross Validation (Hybrid) - Random Search - {save_directory}")
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
    #smote = SMOTE(random_state=48)
    #X_resampled, y_resampled = smote.fit_resample(X, y)

    # Apply undersampling to the entire dataset before RandomizedSearchCV
    #undersampler = RandomUnderSampler(random_state=48)
    #X_resampled, y_resampled = undersampler.fit_resample(X, y)

    X_resampled = X
    y_resampled = y

    ############################################################
    ############################################################
    ############################################################

    # Define the XGBClassifier directly (no pipeline)
    xgb_model = XGBClassifier(
        random_state=48,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Initialize RandomizedSearchCV with 10-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of random combinations to try
        scoring='f1',  # Optimize for F1 score
        cv=KFold(n_splits=10, shuffle=True, random_state=48),
        random_state=48,
        n_jobs=2
#        n_jobs=-1     Teve de ser reduzido para não ficar sem espaço em disco por causa de ficheiros temporários.
    )
    
    # Run RandomizedSearchCV on the undersampled training set
    random_search.fit(X_resampled, y_resampled)
    
    # Output the best hyperparameters
    best_params = random_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Initialize lists to store scores for each fold
    train_accuracy_scores = []
    test_accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []

    # 10-fold cross-validation loop to calculate metrics
    kf = KFold(n_splits=10, shuffle=True, random_state=48)
    for train_index, test_index in kf.split(X_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        
        # Train with the best found hyperparameters
        best_model = XGBClassifier(
            n_estimators=random_search.best_params_['n_estimators'],
            max_depth=random_search.best_params_['max_depth'],
            learning_rate=random_search.best_params_['learning_rate'],
            subsample=random_search.best_params_['subsample'],
            colsample_bytree=random_search.best_params_['colsample_bytree'],
            random_state=48,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        best_model.fit(X_train, y_train)
        
        # Predict and calculate metrics on the test set
        y_test_pred = best_model.predict(X_test)
        test_accuracy, precision, recall, f1, auc = calculate_metrics(y_test, y_test_pred)
        
        # Calculate training accuracy
        y_train_pred = best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Append metrics for each fold
        train_accuracy_scores.append(train_accuracy)
        test_accuracy_scores.append(test_accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # Print mean metrics
    print(f"Mean Training Accuracy: {np.mean(train_accuracy_scores)}")
    print(f"Mean Test Accuracy: {np.mean(test_accuracy_scores)}")
    print(f"Mean Precision: {np.mean(precision_scores)}")
    print(f"Mean Recall: {np.mean(recall_scores)}")
    print(f"Mean F1 Score: {np.mean(f1_scores)}")
    print(f"Mean AUC: {np.mean(auc_scores)}")


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
