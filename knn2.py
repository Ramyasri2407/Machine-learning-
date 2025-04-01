import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    x1 = np.array(x1)  # Convert x1 to numpy array
    x2 = np.array(x2)  # Convert x2 to numpy array
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Function to plot KNN for regression
def plot_knn_regression(X_train, y_train, X_test, k=3):
    plt.figure(figsize=(8, 6))  # Create a figure for the plot
    
    # Plot the training points (y_train values represent the regression target)
    for i in range(len(X_train)):
        plt.scatter(X_train[i][0], X_train[i][1], color='blue', s=100, label=f'Train Point {y_train[i]}' if i == 0 or i == 3 else "")
    
    # Plot the test points
    for x_test in X_test:
        plt.scatter(x_test[0], x_test[1], color='green', marker='X', s=200, label='Test Point')

        # Calculate the distance from the test point to each training point
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
        
        # Sort the distances and get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]

        # Get the target values (y_train) of the k nearest neighbors
        k_nearest_values = [y_train[i] for i in k_indices]

        # The prediction is the mean of the k nearest target values
        prediction = np.mean(k_nearest_values)

        # Plot lines connecting the test point to the k nearest neighbors
        for i in k_indices:
            plt.plot([x_test[0], X_train[i][0]], [x_test[1], X_train[i][1]], 'k--')  # Dashed black lines
        
        # Display the predicted value for the test point
        plt.text(x_test[0], x_test[1] + 0.1, f'Pred: {prediction:.2f}', color='green', ha='center')

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Nearest Neighbors Regression Visualization')
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

# Example data (regression)
X_train = [[1, 2], [2, 3], [3, 1], [6, 7], [7, 8], [8, 6]]  # Training features
y_train = [10, 15, 12, 25, 30, 22]  # Continuous target values (regression labels)
X_test = [[5, 5]]  # Test point
k = 3  # Number of nearest neighbors

# Call the function to plot KNN for regression
plot_knn_regression(X_train, y_train, X_test, k)
