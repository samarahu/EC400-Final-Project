from gym.envs.registration import register

register(
    id='pystk-v0',
    entry_point='gym_pystk.envs:PystkEnv',
)
register(
    # id='foo-extrahard-v0',
    # entry_point='gym_foo.envs:FooExtraHardEnv',
)