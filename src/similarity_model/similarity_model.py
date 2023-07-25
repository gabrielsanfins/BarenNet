from typing import List, Dict
import numpy as np

from .group_calculations import (
    find_buckingham_group_exponents_from_construction_dict)

from .barennet import create_barennet

from utils import adjust_dataframe_according_to_similarity


class SimilarityModel:

    def __init__(self,
                 data_path: str,
                 dimensionally_independent_params: List[str],
                 dimensionally_dependent_params: List[str],
                 dimensional_qoi: List[str],
                 non_dimensional_qoi: List[str],
                 non_dimensional_params_construction:
                 Dict[str, Dict[str, float]],
                 non_dimesional_qoi_construction: Dict[str, Dict[str, float]],
                 non_similar_params: List[str],
                 similar_params: List[str]) -> None:

        if (len(non_dimensional_params_construction) !=
                len(dimensionally_dependent_params)):
            raise ValueError("The non-dimensional construction dictionary "
                             "should have the same length as the "
                             "dimensionally dependent parameters list.")

        self.n_nonsimilar = len(non_similar_params)
        self.n_similar = len(similar_params)

        if (self.n_nonsimilar +
                self.n_similar != len(dimensionally_dependent_params)):
            raise ValueError("The quantity of similar and non-similar"
                             "parameters should be equal to the lenght of the "
                             "dimensionally dependent parameters list.")

        self.data_path = data_path

        if data_path == "":
            pass
        else:
            self.df_log_x, self.df_log_y = (
                adjust_dataframe_according_to_similarity(
                    data_path=data_path,
                    non_similar_params=non_similar_params,
                    similar_params=similar_params,
                    non_dimensional_qoi=non_dimensional_qoi
                )
            )

        self.dimensionally_independent_params = (
                                            dimensionally_independent_params)
        self.dimensionally_dependent_params = dimensionally_dependent_params
        self.dimensional_qoi = dimensional_qoi
        self.non_dimensional_qoi = non_dimensional_qoi
        self.non_dimensional_params_construction = (
                                           non_dimensional_params_construction)
        self.non_dimensional_qoi_construction = non_dimesional_qoi_construction
        self.non_similar_params = non_similar_params
        self.similar_params = similar_params
        self.found_incomplete_similarity = False

        self._create_buckingham_similarity_group()

    def _create_buckingham_similarity_group(self) -> None:
        """

        Function that calculates the Buckingham similarity group from the
        non-dimensional construction provided when an instance of the class is
        created. This method should run every time an instance of the class is
        created in the method __init__.

        """
        similarity_dic = {}
        for nd_key in self.non_dimensional_params_construction.keys():
            dimensional_dict = self.non_dimensional_params_construction[nd_key]
            independent_parameters = self.dimensionally_independent_params
            dependent_parameters = self.dimensionally_dependent_params

            dimensionally_dependent_key, exponents_dict = (
                find_buckingham_group_exponents_from_construction_dict(
                    dimensional_dict=dimensional_dict,
                    dimensionally_independent_params=independent_parameters,
                    dimensionally_dependent_params=dependent_parameters
                )
            )

            similarity_dic[dimensionally_dependent_key] = exponents_dict

        for nd_key in self.non_dimensional_qoi_construction.keys():
            dimensional_dict = self.non_dimensional_qoi_construction[nd_key]
            independent_parameters = self.dimensionally_independent_params
            dependent_parameters = self.dimensional_qoi

            dimensionally_dependent_key, exponents_dict = (
                find_buckingham_group_exponents_from_construction_dict(
                    dimensional_dict=dimensional_dict,
                    dimensionally_independent_params=independent_parameters,
                    dimensionally_dependent_params=dependent_parameters
                )
            )

            similarity_dic[dimensionally_dependent_key] = exponents_dict

        self.buckingham_similarity_group = similarity_dic

    def print_buckingham_similarity_group(self) -> None:
        """

        Prints the Buckingham Similarity Group induced by the non-dimensional
        construction provided when the class is first instanced.

        """
        if self.buckingham_similarity_group is None:
            raise ValueError("There is no Buckingham similarity group to "
                             "print.")

        # First we will build a dictionary which will make a correspodence
        # between the dimensionally independent parameters and their respective
        # group parameters.

        group_params_dict = {}
        di_params = self.dimensionally_independent_params
        for n in range(len(di_params)):
            group_params_dict[di_params[n]] = "A_" + str(n + 1)

        # Print the group transformations for the dimensionally independent
        # parameters.

        for key in group_params_dict.keys():
            print(key + "* = " + group_params_dict[key] + " " + key)

        # Print the group transformations for the dimensionally dependent
        # parameters.

        dd_params = self.dimensionally_dependent_params

        for key in dd_params:
            group_str = self._generate_buckingham_group_str(
                key=key,
                group_params_dict=group_params_dict
            )

            print(key + "* = " + group_str + key)

        # Print the group transformations for the qoi.

        qoi = self.dimensional_qoi

        for key in qoi:
            group_str = self._generate_buckingham_group_str(
                key=key,
                group_params_dict=group_params_dict
            )

            print(key + "* = " + group_str + key)

    def _generate_buckingham_group_str(self,
                                       key: str,
                                       group_params_dict: Dict[str, str]
                                       ) -> str:
        """

        Generates a string to help the generation of the buckingham similarity
        group print.

        """
        group_str = ""

        for di_key in self.buckingham_similarity_group[key]:
            exponent = round(self.buckingham_similarity_group[key][di_key],
                             2)
            exponent_str = str(exponent)
            group_str += (group_params_dict[di_key] + "^{" + exponent_str +
                          "} ")

        return group_str

    def find_incomplete_similarity(self,
                                   dense_activation: str = "relu",
                                   loss: str = "mean_squared_error",
                                   optimizer: str = "adam",
                                   n_tries: int = 10,
                                   n_epochs: int = 100,
                                   tol: float = 1e-3) -> None:
        """

        Creates an instance of barennet according to the non-similar and
        similar parameters passed when the model was first instanced. It will
        try to fit the barennet n_tries times and will consider the incomplete
        similarity to exist only if the fit has an error of less than tol.

        dense_activation must be a tensorflow activation function which will be
        used in the dense part of the NN.

        loss must be a tensorflow loss function.
        optimizer must be a tensorflow optimizer.

        """
        xtrain = self.df_log_x
        ytrain = self.df_log_y
        best_loss = 1000000

        n = 1
        while n < n_tries:
            model = create_barennet(
                n_nonsimilar=self.n_nonsimilar,
                n_similar=self.n_similar,
                dense_activation=dense_activation
            )
            model.compile(loss=loss, optimizer=optimizer)
            model.fit(xtrain, ytrain, epochs=n_epochs, verbose=2)

            current_loss = model.evaluate(xtrain, ytrain, verbose=0)

            if current_loss < best_loss:
                best_loss = current_loss

            if current_loss < tol:
                print("Incomplete Similarity Found!")
                print("Error = " + current_loss)
                self.barennet_model = model
                self._save_incomplete_similarity_relation()
                self.found_incomplete_similarity = True
                return None

            n += 1

        self.found_incomplete_similarity = False

        print("Incomplete Similarity was not found!")
        print("However, the best error found was " +
              str(np.round(best_loss, 3)) + ".")
        print("If you feel this error is small enough, consider increasing"
              "n_tries or n_epochs, or decreasing the tolerance.")

        return None

    def _save_incomplete_similarity_relation(self) -> None:
        """

        Saves the incomplete similarity relation found through barennet. In
        order for this method to work, an incomplete similarity relation should
        have been found with the method find_incomplete_similarity.

        The exponents of the incomplete similarity relation will be saved in
        the attribute self.incomplete_similarity_exponents_dict.

        """
        if not self.found_incomplete_similarity:
            print("Incomplete similarity was not found. In order for an "
                  "incomplete similarity relation to exist, we first need "
                  "to sucessfully find it with the method "
                  "find_incomplete_similarity.")
            return None

        exponents_dict = {}

        for i in range(len(self.non_similar_params)):
            exponents_dict[self.non_similar_params[i]] = {}
            for j in range(len(self.similar_params)):
                exponents_dict[self.non_similar_params[i]][
                    self.similar_params[j]
                ] = (
                    self.barennet_model.get_layer(
                        'similarity_layer').weights[0][j][i].numpy()
                    )

        for i in range(len(self.non_dimensional_qoi)):
            exponents_dict[self.non_dimensional_qoi[i]] = {}
            for j in range(len(self.similar_params)):
                exponents_dict[self.non_dimensional_qoi[i]][
                    self.similar_params[j]
                ] = (
                    self.barennet_model.get_layer(
                        'multiplication_layer').weights[0][j][i].numpy()
                    )

        self.incomplete_similarity_exponents_dict = exponents_dict

        return None
