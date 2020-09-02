from typing import Dict
from sklearn.linear_model import LogisticRegression as Model
from .base import Base


class LogisticRegression(Base):
    """Wrapper for scikit-learn LogisticRegression Model"""

    def __init__(self, x_train: Dict, y_train: Dict, grid_search_verbose: int = 10,
                 grid_search_n_jobs: int = 1, penalty: str = 'l2', class_weight: str = 'balanced') -> None:
        super().__init__(x_train, y_train, Model, grid_search_verbose, grid_search_n_jobs)
        self.__grid_search_params = dict(C=[10 ** x for x in range(-5, 4)], penalty=penalty, class_weight=class_weight)

    @property
    def grid_search_params(self):
        return self.__grid_search_params
