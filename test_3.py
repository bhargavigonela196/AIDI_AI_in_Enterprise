import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Load the Wine dataset
wine_data = load_wine()
X, y = wine_data.data, wine_data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=25)

# Define hyperparameter grid for tuning Random Forest
param_grid = {
    'n_estimators': [50, 100],  # Number of trees in the forest
    'max_depth': [3, 5, 10],  # Limit depth of each tree
    'min_samples_split': [2, 5],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2]    # Minimum samples required at a leaf node
}

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=25)

# Apply GridSearchCV with 5-fold cross-validation for hyperparameter tuning
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best estimator from grid search
best_rf = grid_search.best_estimator_

# Apply k-fold cross-validation to evaluate the best model
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Fit the best model on the full training data
best_rf.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = best_rf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Visualize one of the trees in the Random Forest
plt.figure(figsize=(12, 8))
estimator = best_rf.estimators_[0]  # Visualize first tree in the forest
from sklearn.tree import plot_tree

plot_tree(estimator,
          feature_names=wine_data.feature_names,
          class_names=wine_data.target_names.tolist(),  # Convert to list
          filled=True,
          rounded=True)
plt.title('Decision Tree from Random Forest for Wine Classification')
plt.show()