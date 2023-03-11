import typing as tp


def generate_forward_padding(derivative: int, accuracy: int) -> tp.Tuple[int, int]:
    return (0, derivative + accuracy - 1)


def generate_central_padding(derivative: int, accuracy: int) -> tp.Tuple[int, int]:
    left_offset = -((derivative + accuracy - 1) // 2)
    right_offset = (derivative + accuracy - 1) // 2 + 1
    return (abs(left_offset), abs(right_offset - 1))


def generate_backward_padding(derivative: int, accuracy: int) -> tp.Tuple[int, int]:
    return (abs(-(derivative + accuracy - 1)), 0)


def _add_padding_dims(
    padding: tp.Tuple[int, int], ndim: int, axis: int
) -> tp.List[tp.Tuple[int, int]]:
    full_padding = list()
    for i in range(ndim):
        if i == axis:
            full_padding.append(padding)
        else:
            full_padding.append((0, 0))

    return full_padding
