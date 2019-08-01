import json

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
        ###
        # Environment configuration. Get information from arguments or take default if optional
        ###

        # Forest data path (mandatory)
        if "forest_data_path" in kwargs:
            self.forest_data_path = kwargs["forest_data_path"]
        else:
            raise TypeError("Missing mandatory forest_data_path argument on construction.")

        if "num_measurements" in kwargs and kwargs["num_measurements"] % 2 == 0:
            self.numsats = kwargs["num_measurements"]
            self.numpro = self.numsats / 2
        else:
            raise TypeError("Missing mandatory even num_measurements argument on construction.")

        if "max_forest_heights" in kwargs:
            self.max_forest_heights = kwargs["max_forest_heights"]
        else:
            raise TypeError("Missing mandatory max_forest_heights list argument on construction.")

        if "orbit_altitude" in kwargs:
            self.orbit_altitude = kwargs["orbit_altitude"]
        else:
            raise TypeError("Missing mandatory orbit_altitude float argument on construction.")

        if "draw_plot" in kwargs:
            self.draw_plot = kwargs["draw_plot"]
        else:
            self.draw_plot = True


        # TODO: Compute PRO parameters
        self.wavelength = 0.24
        self.vertres = [max_height/self.numsats for max_height in self.max_forest_heights]
        self.baselines = [self.orbit_altitude*self.wavelength/2/vertres for vertres in self.vertres]
        self.maxdists = [self.orbit_altitude*self.wavelength/2/max_height for max_height in self.max_forest_heights]

        # Information about current state that needs saving
        self.current_formation = 0
        self.target_formation = 0
        self.steps_before_change = 0
        self.num_maneuvers = 100  # Depends on fuel, better change to a better proxy such as max deltaV

        self.action_space = spaces.Discrete(len(self.max_forest_heights)+1)  # 0: Do nothing; 1-6: Give order to change formation ASAP
        # State is: [lon, lat, target_baseline, target_maxdist, remaining_fuel]
        low = np.array([
            -np.pi,
            -np.pi/2,
            0,
            0,
            0
        ])
        high = np.array([
            np.pi,
            np.pi/2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            self.num_maneuvers])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        ###
        # Rendering initialization
        ###

        self.figure = None
        self.earth_axes = None
        self.reward_axes = None
        self.baseline_axes = None
        self.trajectory_axes = None

        self.ground_track_point = None
        self.reward_line = None
        self.fov_footprint_polygon = None
        self.current_baseline_line = None
        self.target_baseline_line = None
        self.pro_ellipses = None

        self.ground_track = {}
        self.fov_footprint = []
        self.reward_timeline = []
        self.current_baseline_timeline = []
        self.target_baseline_timeline = []

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def start_plot(self):
        plt.ion()
        self.figure = plt.figure(constrained_layout=True, figsize=(20, 10))
        widths = [1, 1, 1]
        heights = [2, 1]
        gs = self.figure.add_gridspec(ncols=3, nrows=2, width_ratios=widths, height_ratios=heights)
        self.earth_axes = self.figure.add_subplot(gs[0, :])
        self.earth_axes.set_title('Ground track and footprint')
        self.reward_axes = self.figure.add_subplot(gs[1, 0])
        self.reward_axes.set_title('Reward')
        self.baseline_axes = self.figure.add_subplot(gs[1, 1])
        self.baseline_axes.set_title('Baseline')
        self.trajectory_axes = self.figure.add_subplot(gs[1, 2])
        self.trajectory_axes.set_title('Trajectory')
        plt.show()

        path = geopandas.datasets.get_path('naturalearth_lowres')
        earth_info = geopandas.read_file(path)
        forest_data_path = "/Users/anmartin/Projects/summer_project/gym-orekit/forest_data.tiff"
        src = rasterio.open(forest_data_path)
        show(src, cmap="Greens", ax=self.earth_axes)
        earth_info.plot(ax=self.earth_axes, facecolor='none', edgecolor='black')

    def update_plot(self):
        # Earth plot
        if self.ground_track_point is None:
            self.ground_track_point = self.earth_axes.plot(self.ground_track["lon"], self.ground_track["lat"], marker='.', color='r')[0]
        self.ground_track_point.set_data(self.ground_track["lon"], self.ground_track["lat"])

        if self.fov_footprint_polygon is None:
            self.fov_footprint_polygon = \
                self.earth_axes.plot([point["lon"] for point in self.fov_footprint],
                                [point["lat"] for point in self.fov_footprint])[
                    0]
        self.fov_footprint_polygon.set_data([point["lon"] for point in self.fov_footprint],
                                            [point["lat"] for point in self.fov_footprint])

        # Reward plot
        if self.reward_line is None:
            self.reward_line = self.reward_axes.plot(self.reward_timeline)[0]
        self.reward_line.set_data(range(len(self.reward_timeline)), self.reward_timeline)
        self.reward_axes.relim()
        self.reward_axes.autoscale_view()

        # Baseline plot
        if self.target_baseline_line is None:
            self.target_baseline_line = self.baseline_axes.plot(self.target_baseline_timeline)[0]
        if self.current_baseline_line is None:
            self.current_baseline_line = self.baseline_axes.plot(self.current_baseline_timeline)[0]
        self.target_baseline_line.set_data(range(len(self.target_baseline_timeline)), self.target_baseline_timeline)
        self.current_baseline_line.set_data(range(len(self.current_baseline_timeline)), self.current_baseline_timeline)
        self.baseline_axes.relim()
        self.baseline_axes.autoscale_view()

        # Trajectories plot

        # Animation
        self.figure.canvas.draw_idle()
        self.figure.canvas.start_event_loop(0.001)

    def close(self):
        pass


class OnlineOrekitEnv(OrekitEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ###
        # Initialize server connection
        ###
        port = 9090
        # Make socket
        self.transport = TSocket.TSocket('localhost', port)
        # Buffering is critical. Raw sockets are very slow
        self.transport = TTransport.TBufferedTransport(self.transport)
        # Wrap in a protocol
        self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        # Create a client to use the protocol encoder
        self.client = Orekit.Client(self.protocol)

    def step(self, action):
        # Connect
        self.transport.open()

        # Apply action
        if action != 0 and self.num_maneuvers > 0:
            self.num_maneuvers -= 1
            self.steps_before_change = 90
            self.target_formation = action - 1
            self.client.sendHighLevelCommand(action - 1)

        if self.steps_before_change > 0:
            self.steps_before_change -= 1
            if self.steps_before_change == 0:
                self.current_formation = self.target_formation

        # Advance one step in java
        self.client.step()

        states = self.client.currentStates()
        reward = self.client.getReward()
        done = self.client.done()
        ground_pos = self.client.groundPosition()
        fov_footprint = self.client.getFOV()

        # Close
        self.transport.close()

        state_list = [
            ground_pos.longitude,
            ground_pos.latitude,
            self.baselines[self.current_formation],
            self.maxdists[self.current_formation],
            self.num_maneuvers
        ]

        self.ground_track = {"lon": ground_pos.longitude, "lat": ground_pos.latitude}
        self.fov_footprint = [{"lon": point.longitude, "lat": point.latitude} for point in fov_footprint]
        self.reward_timeline.append(self.reward_timeline[-1] + reward)
        self.current_baseline_timeline.append(self.baselines[self.current_formation])
        self.target_baseline_timeline.append(self.baselines[self.target_formation])

        self.update_plot()

        return np.array(state_list), reward, done, {}

    def reset(self):
        # Java reset
        # Connect
        self.transport.open()

        # Send action to java
        self.client.reset()

        states = self.client.currentStates()
        ground_pos = self.client.groundPosition()

        # Close
        self.transport.close()

        # Python reset
        self.current_formation = 0
        self.target_formation = 0
        self.steps_before_change = 0
        self.num_maneuvers = 100

        self.reward_timeline = [0]
        self.current_baseline_timeline = [self.baselines[self.current_formation]]
        self.target_baseline_timeline = [self.baselines[self.target_formation]]

        state_list = [
            ground_pos.longitude,
            ground_pos.latitude,
            self.baselines[self.current_formation],
            self.maxdists[self.current_formation],
            self.num_maneuvers
        ]

        self.start_plot()

        return np.array(state_list)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class OfflineOrekitEnv(OrekitEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ###
        # Load the json file with the simulation results
        ###
        if "simulation_data_path" in kwargs:
            self.simulation_data_path = kwargs["simulation_data_path"]
        else:
            raise TypeError("Missing mandatory simulation_data_path argument on construction.")

        with open(self.simulation_data) as simulation_file:
            self.simulation_data = json.load(simulation_file)

        self.current_step = 0

    def step(self, action):
        if self.current_step >= len(self.simulation_data["timesteps"]):
            return np.zeros(5), 0, True, {}

        # Apply action
        if self.steps_before_change > 0:
            self.steps_before_change -= 1
            if self.steps_before_change == 0:
                self.current_formation = self.target_formation

        if action != 0 and self.num_maneuvers > 0:
            if action != self.current_formation:
                self.num_maneuvers -= 1
                self.steps_before_change = 90
                self.target_formation = action - 1

        # Advance one step
        # Compute reward
        reward = 0
        for forest in self.simulation_data["timesteps"][self.current_step]["visitedPoints"]:
            if forest == self.current_formation and self.steps_before_change == 0:
                reward += 1
        ground_pos = self.simulation_data["timesteps"][self.current_step]["groundTrack"]
        # fov_footprint = self.client.getFOV() # TODO: Implement in offline

        # Close
        self.transport.close()

        state_list = [
            np.degrees(ground_pos["longitude"]),
            np.degrees(ground_pos["latitude"]),
            self.baselines[self.current_formation],
            self.maxdists[self.current_formation],
            self.num_maneuvers
        ]

        self.ground_track = {"lon": ground_pos.longitude, "lat": ground_pos.latitude}
        self.fov_footprint = []  # TODO: Implement
        self.reward_timeline.append(self.reward_timeline[-1] + reward)
        self.current_baseline_timeline.append(self.baselines[self.current_formation])
        self.target_baseline_timeline.append(self.baselines[self.current_formation])

        self.update_plot()

        self.current_step += 1

        return np.array(state_list), reward, False, {}

    def reset(self):
        self.current_step = 0

        # Python reset
        self.current_formation = 0
        self.target_formation = 0
        self.steps_before_change = 0
        self.num_maneuvers = 100

        self.reward_timeline = [0]
        self.current_baseline_timeline = [self.baselines[self.current_formation]]
        self.target_baseline_timeline = [self.baselines[self.target_formation]]

        ground_pos = self.simulation_data["timesteps"][0]["groundTrack"]

        state_list = [
            np.degrees(ground_pos["longitude"]),
            np.degrees(ground_pos["latitude"]),
            self.baselines[self.current_formation],
            self.maxdists[self.current_formation],
            self.num_maneuvers
        ]

        self.start_plot()

        return np.array(state_list)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
