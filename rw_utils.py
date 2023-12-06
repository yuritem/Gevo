import os
import pickle
import numpy as np
from typing import Union


def serialize_boolean_array(array: np.array) -> bytes:
    """
    Adapted from https://stackoverflow.com/a/48486714

    Takes a np.ndarray with boolean values and converts it to a space-efficient
    binary representation.
    """
    return np.packbits(array).tobytes()


def deserialize_boolean_array(serialized_array: bytes, shape: tuple) -> np.ndarray:
    """
    Adapted from https://stackoverflow.com/a/48486714

    Inverse of serialize_boolean_array.
    """
    num_elements = np.prod(shape)
    packed_bits = np.frombuffer(serialized_array, dtype='uint8')
    result = np.unpackbits(packed_bits)[:num_elements]
    result.shape = shape
    return result


def pickle_obj(obj, filename, path: Union[str, bytes, os.PathLike]) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    if not filename.endswith('.pickle'):
        filename = f'{filename}.pickle'
    with open(os.path.join(path, filename), 'wb') as wf:
        pickle.dump(obj, file=wf, protocol=pickle.DEFAULT_PROTOCOL)


def read_pickled(filename: str, path: Union[str, bytes, os.PathLike]) -> object:
    if not filename.endswith('.pickle'):
        filename = f'{filename}.pickle'
    with open(os.path.join(path, filename), 'rb') as rf:
        obj = pickle.load(rf)
    return obj
