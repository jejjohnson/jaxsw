import typing as tp


def generate_forward_padding(derivative: int, accuracy: int)-> tp.Tuple[int, int]:
    return (0, derivative + accuracy-1)


def generate_central_padding(derivative: int, accuracy: int)-> tp.Tuple[int, int]:
    left_offset = -((derivative + accuracy - 1) // 2)
    right_offset = (derivative + accuracy - 1) // 2 + 1
    return (abs(left_offset), abs(right_offset-1))


def generate_backward_padding(derivative: int, accuracy: int)-> tp.Tuple[int, int]:
    return (abs(-(derivative + accuracy - 1)), 0)
