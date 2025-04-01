import numpy as np 
 
def standardize_data(X): 
     
    Xmeaned = X - np.mean(X) 
    return Xmeaned 
 
def put_covariance_matrix(X): 
    return np.cov(X.T) 
 
def put_eigenvectors_and_values(cov_matrix): 
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix) 
    return eigen_values, eigen_vectors 
 
def sort_eigenvectors(eigen_values, eigen_vectors): 
    sorted_indices = np.argsort(eigen_values)[::-1] 
    sorted_eigenvalues = eigen_values[sorted_indices] 
    sorted_eigenvectors = eigen_vectors[:, sorted_indices] 
    return sorted_eigenvalues, sorted_eigenvectors 
 
def project_data(X, eigenvectors, num_components): 
    selected_eigenvectors = eigenvectors[:, :num_components] 
    return np.dot(X, selected_eigenvectors) 
 
 
def pca(X, num_components): 
    X_standardized = standardize_data(X) 
     
 
    cov_matrix = put_covariance_matrix(X_standardized) 
     
  
    eigenvalues, eigenvectors = put_eigenvectors_and_values(cov_matrix) 
     
    
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvectors(eigenvalues, eigenvectors) 
     
 
    X_reduced = project_data(X_standardized, sorted_eigenvectors, 
num_components) 
     
    return X_reduced, sorted_eigenvalues 
 
X = np.array([[2, 2.4], 
              [0.5, 0.7], 
              [2.2, 2.9], 
              [1.9, 2.2], 
              [3.1, 3.0], 
              [2.3, 2.7], 
[2, 1.6], 
[1, 1.1], 
[1.5, 1.6], 
[1.1, 0.9]]) 
num_components = 2   
X_reduced, eigenvalues = pca(X, num_components) 
print("Reduced Data:") 
print(X_reduced) 
print("Eigenvalues (sorted):") 
print(eigenvalues)
