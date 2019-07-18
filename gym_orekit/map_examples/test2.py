from gym_orekit.envs import OrekitEnv
import matplotlib.pyplot as plt

test_env = OrekitEnv()

test_env.reset()
for i in range(2000):
    state, reward, done, _ = test_env.step(0)
    test_env.render()