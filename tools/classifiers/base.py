from typing import Dict

from sklearn.model_selection import GridSearchCV


class Base:
    """Base class for classification model"""

    def __init__(self, x_train: Dict, y_train: Dict, model: any, grid_search_verbose: int = 10,
                 grid_search_n_jobs: int = 1) -> None:
        """Init function for Base class"""
        self.__x_train = x_train
        self.__y_train = y_train
        self.__model = model()
        self.__model_type = model
        self.__grid_search_verbose = grid_search_verbose
        self.__grid_search_n_jobs = grid_search_n_jobs
        self.__grid_search_params = dict()
        self.__optimal_parameters = None

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def grid_search_verbose(self):
        return self.__grid_search_verbose

    @property
    def grid_search_n_jobs(self):
        return self.__grid_search_n_jobs

    @property
    def model(self):
        return self.__model

    @property
    def grid_search_params(self):
        return self.__grid_search_params

    @property
    def optimal_parameters(self):
        return self.__optimal_parameters

    def fit(self):
        """Method for fit model"""
        return self.__model.fit(self.x_train, self.y_train)

    def find_optimal_params(self, params: Dict) -> None:
        """Find optimal params for model"""
        if not self.optimal_parameters:
            model = self.__model_type()
            grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", n_jobs=self.grid_search_n_jobs,
                                iid=False, verbose=2)
            grid.fit(self.x_train, self.y_train)
            self.__optimal_parameters = grid.best_params_
            print(grid.best_params_)
            self.__model = self.__model_type(n_estimators=10)
        return self.optimal_parameters
