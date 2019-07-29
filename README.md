# Orekit Gym Environment

This is the Orekit Gym Environment to be used with the most popular DeepRL libraries such as OpenAI Spinup, OpenAI Baselines, RLlib, etc.

## Requirements

This requires the user to be running the last version of the Java Orekit Thrift Server. For more instructions on how to build and run the server, see the README.md on that project.

This project also requires a Python 3 distribution, preferably Anaconda, as this is where the system has been tested on a Mac OS X 10.14.6.

## Installation

To install the environment on the local machine, the user should run `pip install -e gym-orekit` on the folder on top of this repository.

## Usage

You can create an instance of the environment with `gym.make('gym_orekit:orekit-v0')`.

*Important*: Make sure the Orekit Thrift Server is running when executing the ML algorithm.

## Configuration

_Work in progress_.

When creating the environment, the user can pass parameters to configure the simulation.

The list of available parameters is the following:

* `forest_data_path`: This is the path to the forest data GeoTIFF file for the system. It is a *required* parameter. For instructions on generating it, check the data-processing repository.

TODO: Change number of satellites, altitude of orbit, fuel available, requirements for each kind of forest, baseline angle, swath, orbital configuration (maybe for asteroids, other complex systems?)