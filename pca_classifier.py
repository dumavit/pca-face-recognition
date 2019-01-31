import numpy as np


class PCAClassifier:
    def __init__(self):
        self.mean = None
        self.X_centered = None
        self.num_components = None
        self.theta = None
        self.eigenvectors = None

    def _update_theta(self, num_components):
        # Update theta only if number of PCA components has been changed or theta is unset
        if self.theta is None or self.num_components != num_components:
            if num_components:
                self.num_components = num_components
            else:
                # NOTE: Number of PCA components is not set
                # Set to number of elements by default
                self.num_components = self.X_centered.shape[1]

            sigma = self.U[:, 0:self.num_components].T @ self.X_centered[:, 0:self.num_components]
            distances = []

            for i in range(self.num_components):
                for j in range(i, self.num_components):
                    distances.append(np.abs(sigma[i] - sigma[j]))
            self.theta = np.max(distances) / 2

    def fit(self, X):
        if len(X) == 0:
            raise ValueError('X cannot be empty')

        X = np.array([x.reshape(-1) for x in X]).T
        self.mean = np.mean(X, 1)
        self.X_centered = np.array([(x - self.mean) for x in X.T]).T

        C = self.X_centered.T @ self.X_centered
        w, v = np.linalg.eig(C)

        eigenvalues_order = np.argsort(w)[::-1][:self.num_components].tolist()

        self.eigenvectors = v[eigenvalues_order]
        U = self.X_centered[:, eigenvalues_order] @ self.eigenvectors
        # Normalize eigenfaces
        self.U = np.array([u_i / np.linalg.norm(u_i) for u_i in U.T]).T

    def reconstruct(self, y, num_components=None):
        self._update_theta(num_components)
        U = self.U[:, 0:self.num_components]
        return U @ (U.T @ (y.reshape(-1) - self.mean)) + self.mean

    def score(self, y, num_components=None):
        return np.linalg.norm(y.reshape(-1) - self.reconstruct(y, num_components))

    def predict(self, y, num_components=None):
        # NOTE: If num_components is changed then theta will be updated
        score = self.score(y, num_components)
        return score < self.theta
