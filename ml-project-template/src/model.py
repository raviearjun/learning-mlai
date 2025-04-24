## Directory: src/model.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self
    def transform(self, X):
        return X - self.mean_face

def build_pipeline():
    return Pipeline([
        ('centering', MeanCentering()),
        ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
        ('svc', SVC(kernel='linear', random_state=177))
    ])

