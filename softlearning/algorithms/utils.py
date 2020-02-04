from copy import deepcopy
from dotmap import DotMap
from collections import OrderedDict      


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm

def create_MVE_algorithm(variant, *args, **kwargs):
    from .mve_sac import MVESAC

    algorithm = MVESAC(*args, **kwargs)

    return algorithm

def create_MBPO_algorithm(variant, *args, **kwargs):
    from mbpo.algorithms.mbpo import MBPO

    algorithm = MBPO(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'SQL': create_SQL_algorithm,
    'MBPO': create_MBPO_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    # @anyboby, workaround for local_example_debug mode, for some reason gets DotMap isntead of 
    # OrderedDict as algorithm_kwargs, which doesn't seem to work for double asteriks !
    if isinstance(algorithm_kwargs, DotMap): 
        algorithm_kwargs = algorithm_kwargs.toDict()

    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
