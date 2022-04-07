# author  : jerrylee
# data    : 2022/2/15
# time    : 18:47
# encoding: utf-8
from pykalman import KalmanFilter


def Kalman1D(observations, damping):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state


if __name__ == '__main__':
    pred = Kalman1D([12, 12, 14, 14, 15], 0.001)
    print(pred)