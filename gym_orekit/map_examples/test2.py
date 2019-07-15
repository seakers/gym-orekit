from gym_orekit.envs import OrekitEnv

test_env = OrekitEnv()

test_env.reset()
for i in range(100):
    test_env.step(0)
    test_env.render()