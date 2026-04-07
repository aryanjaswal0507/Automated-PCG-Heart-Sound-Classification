from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def get_classifiers():
    """Returns a dictionary of classifiers for model comparison."""
    return {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }

def evaluate_models(X, y, cv=10):
    """Evaluates multiple classifiers using stratified K-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    models = get_classifiers()
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        results[name] = {
            'accuracy_mean': np.mean(scores),
            'accuracy_std': np.std(scores)
        }
        
    return results

def train_best_model(X, y, classifier_name='RandomForest', **params):
    """Trains the best-performing model on the full feature set."""
    models = get_classifiers()
    if classifier_name not in models:
        raise ValueError(f"Classifier {classifier_name} not found.")
        
    model = models[classifier_name]
    if params:
        model.set_params(**params)
        
    model.fit(X, y)
    return model
