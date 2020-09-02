from typing import Dict
from sklearn.ensemble import RandomForestClassifier as Model
from .base import Base


class RandomForestClassifier(Base):
    """Wrapper for scikit-learn RandomForestClassifier Model"""

    def __init__(self, x_train: Dict, y_train: Dict, grid_search_verbose: int = 10, grid_search_n_jobs: int = 1,
                 random_state: int = 42) -> None:
        super().__init__(x_train, y_train, Model, grid_search_verbose, grid_search_n_jobs)
        self.__grid_search_params = dict(n_estimators=[10, 50, 100, 500, 1000, 2000, 3000])

    @property
    def grid_search_params(self):
        return self.__grid_search_params
