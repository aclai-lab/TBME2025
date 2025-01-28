import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pygad
import numpy as np
import joblib
import matplotlib.pyplot as plt  # For saving the fitness plot

result_folder = 'ResidualProducts'

df = pd.read_csv(f'{result_folder}/ID-Features-Vote.csv')

# Define the bin thresholds for the specific bin
bin_thresholds = [(0, 11), (42, 50)]

# Initialize a list to store metrics for all iterations
metrics_list = []

# Number of iterations
num_iterations = 10

# Process the bin for each iteration
for iteration in range(num_iterations):
    # Binarize the target variable based on thresholds
    bin_data = df.copy()
    bin_data['vote'] = np.where(
        (bin_data['vote'] >= bin_thresholds[0][0]) & (bin_data['vote'] <= bin_thresholds[0][1]), 0,
        np.where(
            (bin_data['vote'] >= bin_thresholds[1][0]) & (bin_data['vote'] <= bin_thresholds[1][1]), 1, np.nan
        )
    )
    
    # Drop rows where the target is NaN (outside the defined bins)
    bin_data = bin_data.dropna(subset=['vote'])
    
    # Remove unnecessary columns
    bin_data = bin_data.drop(columns=['ID', 'PAINTING'])
    
    # Separate features (X) and target (y)
    X = bin_data.drop(columns=['vote'])
    y = bin_data['vote']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 + iteration)  # Change random state for each iteration
    
    # Define the fitness function for the genetic algorithm
    def fitness_function(ga_instance, solution, solution_idx):
        selected_features = np.where(solution == 1)[0]
        if len(selected_features) == 0:
            return 0  # Avoid empty selections
        
        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_selected, y_train)
        predictions = model.predict(X_test_selected)
        return accuracy_score(y_test, predictions)
    
    # Set up the genetic algorithm
    num_generations = 100
    num_parents_mating = 5
    sol_per_pop = 20
    num_genes = X_train.shape[1]
    
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=[0, 1]
    )
    
    # Run the genetic algorithm
    ga_instance.run()
    
    # Get the best solution (selected features)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    selected_features_indices = np.where(solution == 1)[0]
    selected_features = X.columns[selected_features_indices]
    
    # Convert selected features to a comma-separated string
    selected_features_str = ", ".join(selected_features)
    
    # Train the final model using the selected features
    X_train_selected = X_train.iloc[:, selected_features_indices]
    X_test_selected = X_test.iloc[:, selected_features_indices]
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train_selected, y_train)
    
    # Save the final model to a file
    model_filename = f'{result_folder}/random_forest_model_iteration_{iteration + 1}.pkl'
    joblib.dump(final_model, model_filename)
    
    # Evaluate the final model
    y_pred = final_model.predict(X_test_selected)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Extract values from the confusion matrix
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Calculate Sensitivity, Specificity, PPV, and NPV
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    
    # Store metrics in a dictionary
    metrics = {
        "iteration": iteration + 1,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "selected_features": selected_features_str,  # Add selected features as a string
        "model_filename": model_filename  # Add the model filename
    }
    
    # Append metrics to the list
    metrics_list.append(metrics)
    
    # Print metrics for the current iteration
    print(f"Metrics for iteration {iteration + 1}:")
    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"PPV: {ppv}")
    print(f"NPV: {npv}")
    print("Selected Features:", selected_features_str)
    print("Model saved to:", model_filename)
    print("\n")

# Save all metrics to a CSV file
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f'{result_folder}/iteration_metrics.csv', index=False)

print(f"Metrics saved to '{result_folder}/iteration_metrics.csv'.")