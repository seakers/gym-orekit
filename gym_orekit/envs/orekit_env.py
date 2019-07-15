import gym
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

        self.numsats = 2
        self.numpro = 3
        self.fig = None
        self.ax = None


    def step(self, action):
        # Connect
        self.transport.open()

        # Advance one step in java
        self.client.step()

        state = self.client.currentStates()
        reward = self.client.getReward()
        done = self.client.done()

        # Close
        self.transport.close()

        return np.array([]), reward, done, {}

    def reset(self):
        # Connect
        self.transport.open()

        # Send action to java
        self.client.reset()

        # Close
        self.transport.close()

        self.fig, self.ax = plt.subplots()

        path = geopandas.datasets.get_path('naturalearth_lowres')
        earth_info = geopandas.read_file(path)
        forest_data_path = "/Users/anmartin/Projects/summer_project/gym-orekit/forest_data.tiff"
        src = rasterio.open(forest_data_path)
        show(src, cmap="Greens", ax=self.ax)
        earth_info.plot(ax=self.ax, facecolor='none', edgecolor='black')

        plt.ion()
        plt.show()


    def render(self, mode='human'):
        # Connect
        self.transport.open()
        # Send action to java
        ground_pos = self.client.groundPosition()
        # Close
        self.transport.close()

        print(ground_pos)
        self.ax.plot((ground_pos.longitude), (ground_pos.latitude), marker='.', color='r')
        plt.draw()
        plt.pause(0.001)

    def close(self):
        pass


class Orekit2SatsEnv(OrekitEnv):
    pass


class Orekit4SatsEnv(OrekitEnv):
    pass