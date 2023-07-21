from typing import Dict, List, Tuple


def find_buckingham_group_exponents_from_construction_dict(
        dimensional_dict: Dict[str, float],
        dimensionally_independent_params: List[str],
        dimensionally_dependent_params: List[str]
        ) -> Tuple[str, Dict[str, float]]:
    """

    Function that calculates the buckingham group exponents from any
    non-dimensional construction dictionary and outputs the exponents
    dictionary.

    """
    exponents_dict = {}

    for dimensional_key in dimensional_dict.keys():
        if dimensional_key in dimensionally_dependent_params:
            gamma = dimensional_dict[dimensional_key]
            dimensionally_dependent_key = dimensional_key

    for dimensional_key in dimensional_dict.keys():
        if dimensional_key in dimensionally_independent_params:
            exponents_dict[dimensional_key] = (
                - dimensional_dict[dimensional_key] / gamma)

    return dimensionally_dependent_key, exponents_dict
