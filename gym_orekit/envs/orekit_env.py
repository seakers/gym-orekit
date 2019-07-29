import gym
from gym import spaces
import numpy as np
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from gym_orekit.thrift import Orekit
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import geopandas


class OrekitEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        port = 9090
        # Make socket
        self.transport = TSocket.TSocket('localhost', port)

        # Buffering is critical. Raw sockets are very slow
        self.transport = TTransport.TBufferedTransport(self.transport)

        # Wrap in a protocol
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)

        # Create a client to use the protocol encoder
        self.client = Orekit.Client(self.protocol)

        self.numsats = 4
        self.numpro = 6
        self.nummaneuvers = 100

        self.fig = None
        self.ax = None
        self.ground_track = None
        self.reward_line = None

        self.reward = []

        self.action_space = spaces.Discrete(7)
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


    def step(self, action):
        # Connect
        self.transport.open()

        # Apply action
        if action != 0 and self.nummaneuvers > 0:
            self.nummaneuvers -= 1
            self.client.sendHighLevelCommand(action-1)


        # Advance one step in java
        self.client.step()

        states = self.client.currentStates()
        reward = self.client.getReward()
        done = self.client.done()
        ground_pos = self.client.groundPosition()

        # Close
        self.transport.close()

        state_list = []
        for state in states:
            pos = state.position
            vel = state.velocity
            state_list.extend([pos.x, pos.y, pos.z, vel.x, vel.y, vel.z])

        self.reward.append(self.reward[-1] + reward)

        # Render stuff
        if self.ground_track is None:
            self.ground_track = self.ax[0].plot((ground_pos.longitude), (ground_pos.latitude), marker='.', color='r')[0]
        self.ground_track.set_data((ground_pos.longitude), (ground_pos.latitude))

        if self.reward_line is None:
            self.reward_line = self.ax[1].plot(self.reward)[0]
        self.reward_line.set_data(range(len(self.reward)), self.reward)
        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

        return np.array(state_list), reward, done, {}

    def reset(self):
        # Connect
        self.transport.open()

        # Send action to java
        self.client.reset()

        states = self.client.currentStates()

        # Close
        self.transport.close()

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [2,1]})
        plt.show()

        path = geopandas.datasets.get_path('naturalearth_lowres')
        earth_info = geopandas.read_file(path)
        forest_data_path = "/Users/anmartin/Projects/summer_project/gym-orekit/forest_data.tiff"
        src = rasterio.open(forest_data_path)
        show(src, cmap="Greens", ax=self.ax[0])
        earth_info.plot(ax=self.ax[0], facecolor='none', edgecolor='black')

        self.reward = [0]

        state_list = []
        for state in states:
            pos = state.position
            vel = state.velocity
            state_list.extend([pos.x, pos.y, pos.z, vel.x, vel.y, vel.z])

        return np.array(state_list)

    def render(self, mode='human'):
        plt.ion()

    def close(self):
        pass


class Orekit2SatsEnv(OrekitEnv):
    pass


class Orekit4SatsEnv(OrekitEnv):
    pass