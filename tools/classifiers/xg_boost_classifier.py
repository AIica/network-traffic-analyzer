from typing import Dict

from xgboost import XGBClassifier as Model

from .base import Base


class XGBClassifier(Base):
    """Wrapper for scikit-learn XGBClassifier Model"""

    def __init__(self, x_train: Dict, y_train: Dict, grid_search_verbose: int = 10,
                 grid_search_n_jobs: int = 1) -> None:
        super().__init__(x_train, y_train, Model, grid_search_verbose, grid_search_n_jobs)
        self.__grid_search_params = dict(
            n_estimators=[10, 50, 100, 500, 1000, 2000, 3000],
            learning_rate=[0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            max_depth=[3, 5, 10],
            colsample_bytree=[0.1, 0.3, 0.5, 1],
            subsample=[0.1, 0.3, 0.5, 1])

    @property
    def grid_search_params(self):
        return self.__grid_search_params
