from gym.envs.registration import register

register(
    id='orekit-v0',
    entry_point='gym_orekit.envs:OrekitEnv',
)
register(
    id='orekit-2sats-v0',
    entry_point='gym_orekit.envs:Orekit2SatsEnv',
)
register(
    id='orekit-4sats-v0',
    entry_point='gym_orekit.envs:Orekit4SatsEnv',
)