import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

# Load the glass dataset
columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Target']
glass_data = pd.read_csv("glass.csv", names=columns)

# Split the dataset into features (x) and target (y)
x = glass_data.drop("Target", axis=1)  # Features
y = glass_data["Target"]  # Target

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=2)
rf_classifier.fit(x, y)

# Function to get the feature importance for a specific type of glass
def get_feature_importance_for_type(type_of_glass, randomState):
    rf_classifier = RandomForestClassifier(random_state=randomState)
    rf_classifier.fit(x, y)
    
    # Get the index of the specific type of glass
    glass_index = glass_data[glass_data["Target"] == type_of_glass].index[0]
    
    # Get the feature importance scores for that type of glass
    feature_importance = rf_classifier.feature_importances_
    
    # Get the column names (element names)
    feature_names = x.columns
    
    # Create a DataFrame to store feature importance scores
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    
    # Sort the DataFrame by importance score in descending order
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    # Print the top features (elements) contributing to the specific type of glass
    print(f"Top features contributing to Type {type_of_glass}:")
    print(importance_df)  # Adjust the number as needed
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel('Importance')
    plt.title(f'Feature Importance for Type {type_of_glass}')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important features at the top
    plt.show()
    
    # Return the importance DataFrame
    return importance_df

# Call the function for each type of glass
get_feature_importance_for_type(1,1)
get_feature_importance_for_type(2,2)
get_feature_importance_for_type(3,3)
get_feature_importance_for_type(5,5)
get_feature_importance_for_type(6,6)
get_feature_importance_for_type(7,7)

# Data Preparation
# Check for missing values
missing_Values = glass_data.isnull().sum()

# Check if there are any missing values
if missing_Values.sum() == 0:
    print("No missing values")
else:
    print("Missing Values:\n", missing_Values)

# Explore distribution of glass types
glass_type_counts = glass_data["Target"].value_counts()
print("Distribution of Glass Types:\n", glass_type_counts)

# Feature Selection
# We obviously don't want type of glass (Target) to be on our X axis so we're gonna just drop that
X = glass_data.drop(columns=["Target"])
y = glass_data["Target"]

# Model Building
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(x_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert classification report to a table
report_table = []
for label, metrics in report.items():
    if label.isdigit():
        label = f"Class {label}"
    if isinstance(metrics, dict):
        precision = metrics['precision']
        recall = metrics['recall']
        f1Score = metrics['f1-score']
        support = metrics['support']
    else:
        precision = metrics
        recall = '-'
        f1Score = '-'
        support = '-'
    report_table.append([label, precision, recall, f1Score, support])

# Print the table using tabulate
print(tabulate(report_table, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'], tablefmt='pretty'))


# Determine the most common type of glass
most_common_type = glass_type_counts.idxmax()
print("Most Common Type of Glass:", most_common_type)

# Identify the features' importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=x.columns)
sorted_feature_importance = feature_importances.sort_values(ascending=False)
print("Feature Importances accross the dataset:\n", sorted_feature_importance)

# Plot feature importances
plt.figure(figsize=(11, 7))
sorted_feature_importance.plot(kind='bar')
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
glass_groups = glass_data.groupby("Target")

# Calculate the mean abundances of each element for each type of glass
glass_groups = glass_data.groupby("Target")
element_abundances = glass_groups.mean()

# Print the mean abundances of each element for each type of glass
print("Mean abundances of each element for each type of glass:")
print(element_abundances.to_string())

# Analyse and compare Support Vector Machine to Random Forest
# Initialize models
rf_classifier = RandomForestClassifier(random_state=42)
svm_classifier = SVC(random_state=42)

# Fit models to training data
rf_classifier.fit(x_train, y_train)
svm_classifier.fit(x_train, y_train)


# Make predictions formy forest and my support vector
rf_pred = rf_classifier.predict(x_test)
svm_pred = svm_classifier.predict(x_test)

# Evaluate performance metrics for both model
models = {
    'Random Forest': rf_pred,
    'Support Vector Machine': svm_pred
}

results = {}
for name, pred in models.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, pred, average='weighted')
    f1_score_value = f1_score(y_test, pred, average='weighted')
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1_score': f1_score_value}

# Display performance results
results_df = pd.DataFrame(results)
print("Performance Metrics:")
print(results_df)

# Plot confusion matrix for my random forest
plt.figure(figsize=(8,8))
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", xticklabels=glass_type_counts.index, yticklabels=glass_type_counts.index)
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()