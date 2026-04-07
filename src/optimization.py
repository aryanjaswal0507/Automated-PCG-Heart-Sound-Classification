import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class SpiderMonkeyOptimizer:
    """Class implementing a custom SMO meta-heuristic for hyperparameter tuning."""
    
    def __init__(self, n_pop=20, n_iter=50, random_state=42):
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_solution = None
        self.best_score = -1
        
    def _fitness(self, params, X, y):
        """Calculates the fitness score (cross-validation accuracy) for a given solution."""
        # Mapping continuous parameters to integers for Random Forest
        n_estimators = int(params[0])
        max_depth = int(params[1])
        min_samples_split = int(params[2])
        min_samples_leaf = int(params[3])
        
        # Ensure parameters are within valid ranges
        n_estimators = max(10, min(1000, n_estimators))
        max_depth = max(1, min(50, max_depth))
        min_samples_split = max(2, min(20, min_samples_split))
        min_samples_leaf = max(1, min(20, min_samples_leaf))
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        return np.mean(scores)
        
    def optimize(self, X, y):
        """Performs the optimization Loop."""
        np.random.seed(self.random_state)
        
        # Initial Population: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
        population = np.zeros((self.n_pop, 4))
        population[:, 0] = np.random.randint(10, 501, self.n_pop)  # n_estimators
        population[:, 1] = np.random.randint(1, 41, self.n_pop)   # max_depth
        population[:, 2] = np.random.randint(2, 11, self.n_pop)   # min_samples_split
        population[:, 3] = np.random.randint(1, 11, self.n_pop)   # min_samples_leaf
        
        fitness_scores = np.array([self._fitness(sol, X, y) for sol in population])
        
        self.best_solution = population[np.argmax(fitness_scores)]
        self.best_score = np.max(fitness_scores)
        
        for iteration in range(self.n_iter):
            for i in range(self.n_pop):
                # Update logic (Simplified Mimi-centric approach from original notebook)
                partner = population[np.random.randint(self.n_pop)]
                new_sol = population[i] + np.random.uniform(-1, 1, 4) * (partner - population[i])
                
                # Check performance
                new_score = self._fitness(new_sol, X, y)
                
                if new_score > fitness_scores[i]:
                    population[i] = new_sol
                    fitness_scores[i] = new_score
                    
                    if new_score > self.best_score:
                        self.best_solution = new_sol
                        self.best_score = new_score
                        
        # Return best parameters as a dictionary
        return {
            'n_estimators': int(self.best_solution[0]),
            'max_depth': int(self.best_solution[1]),
            'min_samples_split': int(self.best_solution[2]),
            'min_samples_leaf': int(self.best_solution[3])
        }

def grid_search_tune(X, y):
    """Performs standard Grid Search over common Random Forest hyperparameters."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_
