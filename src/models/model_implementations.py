from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from src.models.base_model import ECGBaseModel

class RandomForestModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', random_state: int = 36,
                 class_weight: str = None): # Dodano class_weight
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight # Zapisano class_weight
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight=self.class_weight # Przekazano class_weight
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class GradientBoostingModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, # Dodano min_samples_leaf
                 subsample: float = 1.0, 
                 max_features: str = None, # Dodano max_features
                 random_state: int = 34):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf # Zapisano min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features # Zapisano max_features
        self.random_state = random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, # Przekazano min_samples_leaf
            subsample=self.subsample,
            max_features=self.max_features, # Przekazano max_features
            random_state=self.random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class SVMModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', 
                 gamma: str = 'scale', # Zmieniono typ gamma na str, aby pasował do 'scale'/'auto'
                 degree: int = 3, # Dodano degree
                 coef0: float = 0.0, # Dodano coef0
                 probability: bool = True, random_state: int = 34):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree # Zapisano degree
        self.coef0 = coef0 # Zapisano coef0
        self.probability = probability
        self.random_state = random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree, # Przekazano degree
            coef0=self.coef0, # Przekazano coef0
            probability=self.probability,
            random_state=self.random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        if not self.probability:
            raise AttributeError("predict_proba is not available when probability=False for SVMModel.")
        return self.model.predict_proba(X)

class LogisticRegressionModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, penalty: str = 'l2', 
                 solver: str = 'liblinear', random_state: int = 34, 
                 max_iter: int = 1000,
                 class_weight: str = None, # Dodano class_weight
                 l1_ratio: float = None): # Dodano l1_ratio
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight # Zapisano class_weight
        self.l1_ratio = l1_ratio # Zapisano l1_ratio
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        current_solver = self.solver
        current_penalty = self.penalty
        current_l1_ratio = self.l1_ratio

        if current_penalty == 'elasticnet':
            if current_solver != 'saga':
                # Informacja o zmianie solvera może być przydatna podczas debugowania
                # print(f"Info: For elasticnet penalty, changing solver from '{current_solver}' to 'saga'.")
                current_solver = 'saga'
            # l1_ratio jest używane tylko z elasticnet
        elif current_penalty == 'l1':
            if current_solver not in ['liblinear', 'saga']:
                # print(f"Info: For l1 penalty, solver '{current_solver}' is not compatible. Changing to 'saga'.")
                current_solver = 'saga' # 'saga' jest bezpiecznym wyborem z dostępnych w SEARCH_SPACES
            current_l1_ratio = None # l1_ratio nie jest używane z 'l1' penalty
        else: # Dla 'l2' lub innych (jeśli zostaną dodane)
            current_l1_ratio = None # l1_ratio nie jest używane z 'l2' penalty

        self.model = LogisticRegression(
            C=self.C,
            penalty=current_penalty,
            solver=current_solver, # Użyj potencjalnie skorygowanego solvera
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            l1_ratio=current_l1_ratio # Przekaż l1_ratio tylko jeśli penalty to elasticnet
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X) 

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class CatBoostModel(ECGBaseModel):
    def __init__(self, iterations: int = 1000, learning_rate: float = 0.1,
                 depth: int = 6, 
                 l2_leaf_reg: float = 3.0, # Dodano l2_leaf_reg
                 random_strength: float = 1.0, # Dodano random_strength
                 bagging_temperature: float = 1.0, # Dodano bagging_temperature
                 border_count: int = 254, # Dodano border_count
                 verbose: bool = False, random_seed: int = 34):
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg # Zapisano l2_leaf_reg
        self.random_strength = random_strength # Zapisano random_strength
        self.bagging_temperature = bagging_temperature # Zapisano bagging_temperature
        self.border_count = border_count # Zapisano border_count
        self.verbose = verbose
        self.random_seed = random_seed 
        self.model = None

    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, # Przekazano l2_leaf_reg
            random_strength=self.random_strength, # Przekazano random_strength
            bagging_temperature=self.bagging_temperature, # Przekazano bagging_temperature
            border_count=self.border_count, # Przekazano border_count
            verbose=self.verbose,
            random_seed=self.random_seed
        )
        self.model.fit(X, y)

    def _predict(self, X: np.ndarray):
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)
    
class XGBoostModel(ECGBaseModel):
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_child_weight: int = 1, # Dodano min_child_weight
                 gamma: float = 0, # Dodano gamma
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0, # Dodano reg_alpha
                 reg_lambda: float = 1, # Dodano reg_lambda
                 random_state: int = 42): 
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight # Zapisano min_child_weight
        self.gamma = gamma # Zapisano gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha # Zapisano reg_alpha
        self.reg_lambda = reg_lambda # Zapisano reg_lambda
        self.random_state = random_state 
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight, # Przekazano min_child_weight
            gamma=self.gamma, # Przekazano gamma
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha, # Przekazano reg_alpha
            reg_lambda=self.reg_lambda, # Przekazano reg_lambda
           # use_label_encoder=False, # Zakomentowane, ponieważ może być przestarzałe lub powodować problemy
            eval_metric='logloss',
            random_state=self.random_state 
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)