import traceback
import functools

# The decorator is used to prints an error trhown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper

import numpy as np
def dense_ind(index_map,stay_shape=True,shuffle=False,return_convert=False):
    mapping = np.unique(index_map).astype('int32')
    map_key = np.arange(len(mapping)).astype('int32')
    if shuffle:
        np.random.shuffle(map_key)
    index_map = convert_array(index_map, mapping, map_key, stay_shape)

    convert = {key:idx for key, idx in zip(map_key, mapping)} if return_convert else {}
    return index_map, convert

def convert_array(index_map, mapping, map_key, stay_shape=True):
    assert len(mapping) == len(map_key)
    mapping = np.array(mapping).astype('int32')
    map_key = np.array(map_key).astype('int32')
    map_key = map_key[np.argsort(mapping)]
    mapping = np.sort(mapping)
    index = np.digitize(index_map.ravel(), mapping, right=True)
    if stay_shape:
        index_map = map_key[index].reshape(index_map.shape)
    else:
        index_map = map_key[index]
    if index_map.max() < 256:
        index_map = index_map.astype(np.uint8)
    return index_map
