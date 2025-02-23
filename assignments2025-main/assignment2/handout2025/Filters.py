import random
import numpy as np

from models import *

class HMMFilter:
    def __init__(self, initial_probs, transition_model, observation_model, state_model):
        self.tm = transition_model  # Transition model (a_ij)
        self.om = observation_model  # Observation model (b_j(y_t))
        self.f = initial_probs  # Initial probability distribution (\pi)
        self.sm = state_model

    def filter(self, sensorR: int) -> np.array:
        """
        Forward algorithm for filtering.
        Args:
            sensorR: Single sensor reading (index!)
        Returns:
            Final filtered probabilities
        """
        # Current belief state
        f = self.f.copy()
        
        # Prediction step: f_{t|t-1} = f_{t-1|t-1} * T
        f = np.dot(f, self.tm.get_T())

        # Update step (always do update, even for "nothing" readings)
        o = self.om.get_o_reading(sensorR)
        f = f * np.diag(o)
        
        # Normalize
        if np.sum(f) > 0:
            f = f / np.sum(f)
        
        # Update belief state
        self.f = f
        
        return f


class HMMSmoother:
    def __init__(self, hmm_filter: HMMFilter):
        self.hmm_filter = hmm_filter
        self.k = 5  # Sequence length for smoothing

    def smooth(self, sensor_r_seq: np.array, f_k: np.array) -> np.array:
        """
        Backward algorithm for smoothing.
        Args:
            sensor_r_seq: sequence with the t-k sensor readings for smoothing
            f_k: filtered result (f_vector) for step k
        Returns:
            Smoothed state probabilities (fb_vector)
        """
        K = len(self.hmm_filter.f)  # Number of states
        
        # Initialize beta with ones for the last time step
        beta = np.ones(K)
        
        # Backward recursion through the sequence
        for t in range(len(sensor_r_seq)-2, -1, -1):
            next_sensor = sensor_r_seq[t+1]
            # Always do update, even for "nothing" readings
            # Get observation probabilities for the next sensor reading
            o = self.hmm_filter.om.get_o_reading(next_sensor)
            # Update beta
            # beta_t = T ^ T * (o_{t+1} * beta_{t+1})
            beta = np.dot(self.hmm_filter.tm.get_T().T, np.diag(o) @ beta)

        # Compute smoothed probabilities
        smoothed = f_k * beta

        if (np.sum(smoothed) > 0):
            smoothed /= np.sum(smoothed)

        # Normalize
        smoothed /= np.sum(smoothed)
        
        return smoothed

class HMMForwardBackward:
    def __init__(self, hmm_filter: HMMFilter, hmm_smoother: HMMSmoother):
        self.hmm_filter = hmm_filter
        self.hmm_smooth = hmm_smoother
        self.sensor_history = []  # Keep track of last 5 sensor readings

    def forward_backward(self, observations: list) -> np.array:
        """
        Forward-backward algorithm for smoothing.
        Minimum number of observations is 5.
        Args:
            observations: List of sensor readings y_1:T
        Returns:
            Smoothed state probabilities
        """
        T = len(observations)
        N = self.hmm_filter.tm.get_num_of_states()
        
        # Forward pass
        alpha = np.zeros((T, N))
        f = self.hmm_filter.f.copy()
        
        for t in range(T):
            # Prediction and update (always do both)
            f = np.dot(f, self.hmm_filter.tm.get_T())
            o = self.hmm_filter.om.get_o_reading(observations[t])
            f = f * np.diag(o)
            # Normalize
            if np.sum(f) > 0:
                f = f / np.sum(f)
            alpha[t] = f
        
        # Backward pass
        beta = np.ones((T, N))
        for t in range(T-2, -1, -1):
            o = self.hmm_filter.om.get_o_reading(observations[t+1])
            beta[t] = np.dot(np.dot(self.hmm_filter.tm.get_T(), o), beta[t+1])

            if np.sum(beta[t]) > 0:
                beta[t] = beta[t] / np.sum(beta[t])
        
        # Combine forward and backward passes
        smoothed = alpha * beta
        # Normalize
        smoothed = smoothed / np.sum(smoothed, axis=1, keepdims=True)
        
        return smoothed