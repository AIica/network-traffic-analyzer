from typing import List, Union

import numpy as np
import pandas as ps
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Model:
    """Base class for models"""

    drop_protos = ["Unknown", "Unencryped_Jabber", "NTP", "Apple"]
    replace_protos = [("SSL_No_Cert", "SSL")]

    def __init__(self, data_train: List, model: any, seed: Union[None, int] = None) -> None:
        """Init Model class"""

        self.__preprocess(data_train, seed)
        self.__model = model
        self.__results = dict()

    def __data_prepare(self, data_train: Union[DataFrame, Series], scale: StandardScaler,
                       labeler: LabelEncoder) -> None:
        """Get data for train. Also scale and labeler for data augmentation"""

        self.__data_train = data_train
        self.__scale = scale
        self.__labeler = labeler

    def __preprocess(self, data: List, seed: int) -> None:
        """Preprocess data. Add scaling"""

        data = data[~data["proto"].isin(self.drop_protos)]
        for old_proto, new_proto in self.replace_protos:
            data = data.replace(old_proto, new_proto)

        if seed:
            np.random.seed(seed)
            data = data.iloc[np.random.permutation(len(data))]

        scale = StandardScaler()
        labeler = LabelEncoder()
        x = scale.fit_transform(data.drop(["proto", "subproto"], axis=1))
        y = labeler.fit_transform(data["proto"])

        cols = [col for col in data.columns if col not in ("proto", "subproto")]
        train_data = ps.concat([ps.DataFrame(x, columns=cols),
                                ps.DataFrame({"proto": y})], axis=1)
        self.__data_prepare(train_data, scale, labeler)

    @staticmethod
    def ___split_data(train_data: Union[DataFrame, Series], clusters_count: int) -> List[Union[DataFrame, Series]]:
        """Split data for train samples"""

        proto_clusters = [train_data[train_data["proto"] == proto] for proto in train_data["proto"].unique()]
        clusters = [[], [], []]
        for cluster in proto_clusters:
            split_index = len(cluster) // clusters_count
            for i in range(clusters_count):
                clusters[i].append(
                    cluster.iloc[i * split_index: (i + 1) * split_index])
        return [ps.concat(clus) for clus in clusters]

    @property
    def results(self):
        return self.__results

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def x_test(self):
        return self.__x_test

    @property
    def y_test(self):
        return self.__y_test

    @property
    def scale(self):
        return self.__scale

    @property
    def labeler(self):
        return self.__labeler

    def train(self):
        """Train model with x_train and y_train"""

        if self.__model is None:
            raise Exception("Model not defined")
        return self.__model.fit(self.x_train, self.y_train)

    def validate(self, model) -> (List, List, List):
        """Validate model and get metrics"""

        y_predicted = model.predict(self.x_test)
        true_labels = self.labeler.inverse_transform(self.y_test)
        predicted_labels = self.labeler.inverse_transform(y_predicted)
        return self.x_test.columns, true_labels, predicted_labels

    def ___data_prepare(self, train_data: Union[DataFrame, Series], test_data: Union[DataFrame, Series]) -> None:
        """Numpy vectors transform"""

        self.__x_train = train_data.drop(["proto"], axis=1)
        self.__y_train = train_data["proto"]
        self.__x_test = test_data.drop(["proto"], axis=1)
        self.__y_test = test_data["proto"]

    def process(self, clusters_count: int = 3) -> None:
        """Main function for model"""
        clusters = self.___split_data(self.__data_train, clusters_count)
        print(f"Data was splited to {clusters_count} clusters")
        for i, train_data in enumerate(clusters):
            test_data = ps.concat([c for c in clusters if c is not self.__data_train])
            self.___data_prepare(train_data, test_data)
            model = self.train()
            self.__results[i] = dict(data=self.validate(model), model=model)
            print(f"Cluster {i + 1} finished")
