from gym.envs.registration import register

register(
    id='orekit-v0',
    entry_point='gym_orekit.envs:OrekitEnv',
)

register(
    id='online-orekit-v0',
    entry_point='gym_orekit.envs:OnlineOrekitEnv',
)

register(
    id='offline-orekit-v0',
    entry_point='gym_orekit.envs:OfflineOrekitEnv',
)