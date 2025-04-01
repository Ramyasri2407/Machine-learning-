import numpy as np
import matplotlib.pyplot as plt
class LDA:
    def fit(self, X, y) :
        self.classlabels = np.unique(y)
        features = X.shape[1]
        classes = len(self.classlabels)
        overall_mean = np.mean(X, axis=0)
        Sw = np.zeros((features, features))
        Sb = np.zeros((features, features))
        for cls in self.classlabels:
            X_cls = X[y == cls]
            mean_cls = np.mean(X_cls, axis=0)
            n_cls = X_cls.shape[0]
            Sw =Sw+np.cov(X_cls, rowvar=False) * (n_cls - 1)
            
            mean_diff = (mean_cls - overall_mean).reshape(features, 1)
            Sb = Sb+n_cls * (mean_diff).dot(mean_diff.T)
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        sorted_indices = np.argsort(eig_vals)[::-1]
        self.scalings = eig_vecs[:, sorted_indices]
        self.eig_vals = eig_vals[sorted_indices]
    def transform(self, X):
        return X.dot(self.scalings)
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [1, 1],  
              [5, 6], [6, 5], [7, 7], [6, 6], [5, 5]]) 
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
lda = LDA()
lda.fit(X,y)
X_lda = lda.transform(X)
for cls in lda.classlabels:
    print(f"Class {cls} LD values: {X_lda[y == cls].flatten()}")
plt.figure(figsize=(10, 6))
colors = ['red', 'blue']
markers = ['o', 's']
for i in range(2):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', color=colors[i], marker=markers[i], s=100)
eig_vec = lda.scalings[:, 0]  
eig_val = lda.eig_vals[0]  
origin = np.mean(X, axis=0)
plt.plot([0, eig_vec[0]*10], [0, eig_vec[1]*10], color='green', linewidth=3, label=f'LD1 (eigenvalue = {eig_val:.4f})')
plt.title(f'LDA projection vector with the highest eigenvalue = {eig_val:.4f}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)      
plt.show()
