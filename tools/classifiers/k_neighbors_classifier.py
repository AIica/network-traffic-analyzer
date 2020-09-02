from typing import Dict
from sklearn.neighbors import KNeighborsClassifier as Model
from .base import Base


class KNeighborsClassifier(Base):
    """Wrapper for scikit-learn KNeighborsClassifier Model"""

    def __init__(self, x_train: Dict, y_train: Dict, grid_search_verbose: int = 10,
                 grid_search_n_jobs: int = 1) -> None:
        super().__init__(x_train, y_train, Model, grid_search_verbose, grid_search_n_jobs)
        self.__grid_search_params = dict(n_neighbors=[i for i in range(1, 15, 2)])

    @property
    def grid_search_params(self):
        return self.__grid_search_params
