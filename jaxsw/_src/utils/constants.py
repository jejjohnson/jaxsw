import math

R_EARTH = 6371200.0  # radius of the Earth (m)
GRAVITY = 9.80665  # gravitational acceleration (m/s^2)
OMEGA = 2.0 * math.pi / 86164.0  # angular speed of the Earth (7.292e-5) (rad/s)
DEG2M = math.pi * R_EARTH / 180.0  # Degrees to Meters
RHO = 1.0e3  # density of water (kg/m^3)
