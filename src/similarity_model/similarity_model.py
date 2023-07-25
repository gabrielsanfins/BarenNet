from typing import List, Dict
import tensorflow as tf

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
