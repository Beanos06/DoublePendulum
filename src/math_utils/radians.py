import math

def simplify_rad(angle: float) -> float:
    """
    Takes in an angle in radians and outputs the same angle between 0 and 2pi
    """

    return angle % 2 * math.pi
