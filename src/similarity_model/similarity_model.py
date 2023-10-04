from typing import List, Dict
import numpy as np

from .group_calculations import (find_buckingham_group_exponents_matrix)

from .barennet import create_barennet

from utils import adjust_dataframe_according_to_similarity


class SimilarityModel:

    def __init__(self,
                 data_path: str,
                 dimensionally_independent_params: List[str],
                 dimensionally_dependent_params: List[str],
                 dimensional_qoi: str,
                 non_dimensional_params: List[str],
                 non_dimensional_qoi: str,
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
        self.non_dimensional_params = non_dimensional_params
        self.non_dimensional_qoi = non_dimensional_qoi
        self.non_dimensional_params_construction = (
                                           non_dimensional_params_construction)
        self.non_dimensional_qoi_construction = non_dimesional_qoi_construction
        self.non_similar_params = non_similar_params
        self.similar_params = similar_params
        self.found_incomplete_similarity = False

        self.A_matrix, self.B_matrix = self._create_exponents_matrices()
        self.Delta_matrix = find_buckingham_group_exponents_matrix(
            self.A_matrix, self.B_matrix)

        self._create_buckingham_similarity_group()

    def _create_exponents_matrices(self):
        """

        Function that the creates the exponents matrices according to the MDDP
        construction theory (see README.md).

        """
        m = len(self.dimensionally_independent_params)
        l = len(self.dimensionally_dependent_params)

        A_matrix = np.zeros(shape=(l, m))
        B_matrix = np.zeros(shape=(l, l))

        for i in range(l):
            for j in range(m):
                A_matrix[i, j] = - self.non_dimensional_params_construction[
                    self.non_dimensional_params[i]][
                        self.dimensionally_independent_params[j]]

            for j in range(l):
                B_matrix[i, j] = self.non_dimensional_params_construction[
                    self.non_dimensional_params[i]][
                        self.dimensionally_dependent_params[j]]

        return A_matrix, B_matrix

    def _create_buckingham_similarity_group(self) -> None:
        """

        Function that calculates the Buckingham similarity group from the
        non-dimensional construction provided when an instance of the class is
        created. This method should run every time an instance of the class is
        created in the method __init__.

        """
        similarity_dict = self._initialize_buckingham_similarity_dict()

        for j in range(len(self.dimensionally_dependent_params)):
            exponents_dict = {}
            for i in range(len(self.dimensionally_independent_params)):
                exponents_dict["A_" + str(i+1)] = self.Delta_matrix[j, i]

            similarity_dict[self.dimensionally_dependent_params[j]] = (
                exponents_dict
            )

        nd_qoi_exponents_dict = {}
        denominator = self.non_dimensional_qoi_construction[
            self.non_dimensional_qoi][self.dimensional_qoi]

        for i in range(len(self.dimensionally_independent_params)):
            alpha = self.non_dimensional_qoi_construction[
                self.non_dimensional_qoi][
                    self.dimensionally_independent_params[i]]

            exponents_sum = 0

            for j in range(len(self.dimensionally_dependent_params)):
                exponents_sum += (self.Delta_matrix[j, i] *
                                  self.non_dimensional_qoi_construction[
                                      self.non_dimensional_qoi][
                                          self.dimensionally_dependent_params[j]
                                      ])

            numerator = alpha + exponents_sum
            exponent = numerator / denominator
            nd_qoi_exponents_dict["A_" + str(i + 1)] = exponent

        similarity_dict[self.dimensional_qoi] = nd_qoi_exponents_dict

        self.buckingham_similarity_group = similarity_dict

    def _initialize_buckingham_similarity_dict(
            self) -> Dict[str, Dict[str, float]]:
        """

        Initializes a dicitionary for the construction of the Buckinham
        similarity group. Only adds the construction of the exponents for the
        dimensionally independent parameters.

        """
        similarity_dict = {}

        for n in range(len(self.dimensionally_independent_params)):
            param_dict = {}
            for m in range(len(self.dimensionally_independent_params)):
                if m == n:
                    param_dict["A_" + str(m + 1)] = 1.0
                else:
                    param_dict["A_" + str(m + 1)] = 0.0

            similarity_dict[self.dimensionally_independent_params[n]] = (
                param_dict
            )
        return similarity_dict

    def print_buckingham_similarity_group(self) -> None:
        """

        Prints the Buckingham Similarity Group induced by the non-dimensional
        construction provided when the class is first instanced.

        """
        if self.buckingham_similarity_group is None:
            raise ValueError("There is no Buckingham similarity group to "
                             "print.")

        for gov_parameter in self.buckingham_similarity_group.keys():
            param_str = gov_parameter + "* = "
            sim_dict = self.buckingham_similarity_group

            for group_parameter in sim_dict[gov_parameter].keys():
                param_str += (group_parameter + "^" +
                              str(sim_dict[gov_parameter][group_parameter]) +
                              " ")

            print(param_str + gov_parameter)

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

        for i in range(len([self.non_dimensional_qoi])):
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
