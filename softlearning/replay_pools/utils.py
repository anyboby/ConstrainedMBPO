from copy import deepcopy

from . import (
    simple_replay_pool,
    extra_policy_info_replay_pool,
    union_pool,
    trajectory_replay_pool,
    mjc_state_replay_pool,
    cpobuffer,
    )


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'TrajectoryReplayPool': trajectory_replay_pool.TrajectoryReplayPool,
    'ExtraPolicyInfoReplayPool': (
        extra_policy_info_replay_pool.ExtraPolicyInfoReplayPool),
    'UnionPool': union_pool.UnionPool,
    'MjcStateReplayPool': mjc_state_replay_pool.MjcStateReplayPool,
    'CPOBuffer': cpobuffer.CPOBuffer,
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_variant(variant, env, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params']
    replay_pool_type = replay_pool_params['type']
    preprocess_type = replay_pool_params['preprocess_type']
    replay_pool_kwargs = deepcopy(replay_pool_params['kwargs'])

    ### check if env has extended action space
    if 'Safexp' in preprocess_type:
        assert hasattr(env, 'action_space_ext')
        action_space = env.action_space_ext
    else: 
        action_space = env.action_space

    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        observation_space=env.observation_space,
        action_space=action_space,  ###@anyboby
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool
