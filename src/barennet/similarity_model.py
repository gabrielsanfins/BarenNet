from typing import List, Dict


class SimilarityModel:

    def __init__(self,
                 data_path: str,
                 dimensionally_independent_params: List[str],
                 dimensionally_dependent_params: List[str],
                 dimensional_qoi: str,
                 non_dimensional_qoi: str,
                 non_dimensional_params_construction:
                 Dict[str, Dict[str, float]],
                 non_dimesional_qoi_construction: Dict[str, Dict[str, float]],
                 n_nonsimilar: int,
                 n_similar: int) -> None:

        if (len(non_dimensional_params_construction) !=
                len(dimensionally_dependent_params)):
            raise ValueError("The non-dimensional construction dictionary "
                             "should have the same length as the "
                             "dimensionally dependent parameters list.")

        if n_nonsimilar + n_similar != len(dimensionally_dependent_params):
            raise ValueError("The quantity of similar and non-similar"
                             "parameters should be equal to the lenght of"
                             "dimensionally dependent parameters list.")

        self.data_path = data_path
        self.dimensionally_independent_params = (
                                            dimensionally_independent_params)
        self.dimensionally_dependent_params = dimensionally_dependent_params
        self.dimensional_qoi = dimensional_qoi
        self.non_dimensional_qoi = non_dimensional_qoi
        self.non_dimensional_params_construction = (
                                           non_dimensional_params_construction)
        self.non_dimensional_qoi_construction = non_dimesional_qoi_construction
        self.n_nonsimilar = n_nonsimilar
        self.n_similar = n_similar

        self.create_buckingham_similarity_group()

    def create_buckingham_similarity_group(self) -> None:
        """

        Function that calculates the Buckingham similarity group from the
        non-dimensional construction provided when an instance of the class is
        created. This method should run every time an instance of the class is
        created in the method __init__.

        """
        similarity_dic = {}
        for nd_key in self.non_dimensional_params_construction.keys():
            dimensional_dic = self.non_dimensional_params_construction[nd_key]
            exponents_dict = {}

            # Firts we find the exponent associated with the dimensionally
            # dependent parameter
            for dimensional_key in dimensional_dic.keys():
                if dimensional_key in self.dimensionally_dependent_params:
                    gamma = dimensional_dic[dimensional_key]
                    dimensionally_dependent_key = dimensional_key

            for dimensional_key in dimensional_dic.keys():
                if dimensional_key in self.dimensionally_independent_params:
                    exponents_dict[dimensional_key] = (
                        - dimensional_dic[dimensional_key] / gamma)

            similarity_dic[dimensionally_dependent_key] = exponents_dict

        self.buckingham_similarity_group = similarity_dic
