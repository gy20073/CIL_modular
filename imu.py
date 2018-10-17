import copy
from datetime import datetime
import os
import numpy as np
from scipy.ndimage.interpolation import shift


class IMU(object):
    def __init__(self):
        NSIZE = 3
        __CARLA_VERSION__  = os.getenv('CARLA_VERSION', '0.8.X')

        self._position = {'x': np.full(NSIZE, np.nan), 'y': np.full(NSIZE, np.nan), 'z': np.full(NSIZE, np.nan)}
        self._orientation = {'pitch': np.full(NSIZE, np.nan), 'roll':np.full(NSIZE, np.nan), 'yaw': np.full(NSIZE, np.nan)}
        self._time = np.full(NSIZE, np.nan)


    def update_state(self, measurements):
        # time update
        shift(self._times, -1, cval=np.NaN)
        self._times[-1] = datetime.now()

        # position update
        self._position['x'] = shift(self._position['x'], -1, cval=np.NaN)
        self._position['x'][-1] = measurements.player_measurements.transform.location.x

        self._position['y'] = shift(self._position['y'], -1, cval=np.NaN)
        self._position['y'][-1] = measurements.player_measurements.transform.location.y

        self._position['z'] = shift(self._position['z'], -1, cval=np.NaN)
        if __CARLA_VERSION__ == '0.8.X':
            self._position['z'][-1] = 1.6
        else:
            self._position['z'][-1] = measurements.player_measurements.transform.location.z

        # orientation update
        self._orientation['pitch'] = shift(self._orientation['pitch'], -1, cval=np.NaN)
        self._orientation['pitch'][-1] = measurements.player_measurements.transform.orientation.x

        self._orientation['roll'] = shift(self._orientation['roll'], -1, cval=np.NaN)
        self._orientation['roll'][-1] = measurements.player_measurements.transform.orientation.y

        self._orientation['yaw'] = shift(self._orientation['yaw'], -1, cval=np.NaN)
        self._orientation['yaw'][-1] = measurements.player_measurements.transform.orientation.z

    def _compute_inertial_measurements(self):
        imu_dict = {'acc_x':0, 'acc_y':0, 'acc_z':0, 'd_pitch':0, 'd_roll':0, 'd_yaw':0, 'dt':0}

        # if not warming-up period
        if not np.any(np.isnan(self._time)):
            delta = self._times[-1] - self._times[-2]
            delta = delta.seconds + delta.microseconds / 1E6
            imu_dict['dt'] = delta

            imu_dict['acc_x'] =  (self._position['x'][-1] - 2.0 * self._position['x'][-2] + self._position['x'][-3]) / (delta * delta)
            imu_dict['acc_y'] =  (self._position['y'][-1] - 2.0 * self._position['y'][-2] + self._position['y'][-3]) / (delta * delta)
            imu_dict['acc_z'] =  (self._position['z'][-1] - 2.0 * self._position['z'][-2] + self._position['z'][-3]) / (delta * delta)

            imu_dict['d_pitch'] = (self._orientation['pitch'][-1] - self._orientation['pitch'][-2]) / delta
            imu_dict['d_roll'] = (self._orientation['roll'][-1] - self._orientation['roll'][-2]) / delta
            imu_dict['d_yaw'] = (self._orientation['yaw'][-1] - self._orientation['yaw'][-2]) / delta

        return imu_dict

    def __call__(self, *args, **kwargs):
        measurements = args[0]
        self.update_state(measurements)
        return self._compute_inertial_measurements()
