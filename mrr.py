def mrr_multiply_bit(input_bit: int, weight_bit: int) -> int:
    """
    Ideal first-stage MRR model.

    In this simplified version, MRR behaves like bitwise AND:
    1 and 1 -> 1
    otherwise -> 0
    """
    if input_bit not in (0, 1) or weight_bit not in (0, 1):
        raise ValueError("MRR bit inputs must be 0 or 1.")

    return input_bit & weight_bit