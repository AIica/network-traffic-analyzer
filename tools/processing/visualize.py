from typing import Dict

import numpy as np
import pandas as ps
from pandas import DataFrame
from sklearn.metrics import classification_report


class Visualize:
    """Class for visualize training results"""

    def __init__(self, model: any, columns: Dict, true_labels: Dict, predicted_labels: Dict) -> None:
        """"Init function"""
        self.__model = model
        self.__columns = columns
        self.__true_labels = true_labels
        self.__predicted_labels = predicted_labels

    def __view_feature_importance(self) -> str:
        """Function for getting feature_importances"""
        imp = {col: imp for imp, col
               in zip(self.__model.feature_importances_, self.__columns)}
        assert len(imp) == len(self.__columns)
        return "\n".join("{}{:.4f}".format(str(col).ljust(25), imp)
                         for col, imp in sorted(imp.items(), key=(lambda x: -x[1])))

    def __view_classification_class_report(self) -> DataFrame:
        """Function for getting classification_class_report"""
        classes = np.unique(self.__true_labels)
        res = ps.DataFrame({"y": self.__true_labels, "p": self.__predicted_labels}, index=None)
        table = ps.DataFrame(index=classes, columns=classes)
        for true_cls in classes:
            tmp = res[res["y"] == true_cls]
            for pred_cls in classes:
                table[pred_cls][true_cls] = len(tmp[tmp["p"] == pred_cls])
        return table

    @property
    def view_classification_report_sklearn(self):
        return self.__view_classification_report_sklearn

    @property
    def view_feature_importance(self):
        return self.__view_feature_importance

    @property
    def view_classification_class_report(self):
        return self.__view_classification_class_report

    def __view_classification_report_sklearn(self) -> None:
        """Function for getting classification_report of sklearn"""
        return classification_report(self.__true_labels, self.__predicted_labels)

    def view_report(self) -> None:
        """Get full report for training process"""
        print(50 * '-')
        print(self.view_feature_importance())
        print(self.view_classification_report_sklearn())
        print(self.view_classification_class_report())
        print(50 * '-')
