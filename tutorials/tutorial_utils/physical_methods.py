def laminar_flow_wall_coordinates(y_plus, re_tau):
    """

    Function that calculates the laminar flow in wall coordinates, i.e. gives
    u^+ as a function of y^+ and Re_tau.

    """

    u_plus = y_plus - ((1.0 / (2.0 * re_tau)) * (y_plus**2))
    return u_plus
