import numpy as np
from hmmlearn.hmm import GaussianHMM

class RegimeDetector:
    """
    Hidden Markov Model for detecting and predicting market regimes.
    
    Identifies latent states (e.g., Calm vs. Crisis) from market features.
    Provides tools for state classification and probability estimation.
    """
    
    def __init__(self, n_states=2, random_state=42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=random_state
        )
        self.state_map = None # Maps internal HMM state to [Low Vol, High Vol]

    def fit(self, features):
        """Fit the HMM to provided features."""
        self.model.fit(features)
        self._map_states(features)
        return self

    def predict_proba(self, features):
        """
        Predict probability of each state for the given features.
        Returns probabilities mapped to [Low Vol, High Vol].
        """
        probas = self.model.predict_proba(features)
        # Reorder probabilities based on our state_map
        return probas[:, self.state_map]

    def get_latest_proba(self, features):
        """Get state probabilities for the most recent observation."""
        probas = self.predict_proba(features)
        return probas[-1]

    def _map_states(self, features):
        """
        Internal helper to identify which HMM state corresponds to 'Low Volatility'.
        We define the 'Low Vol' state (index 0) as the one with the lowest 
        average volatility feature.
        """
        internal_states = self.model.predict(features)
        
        # Calculate mean volatility for each internal state
        # (Assuming volatility is the second feature, index 1)
        state_means = []
        for i in range(self.n_states):
            state_mask = (internal_states == i)
            if state_mask.any():
                # features[:, 1] is volatility
                state_means.append(features[state_mask, 1].mean())
            else:
                state_means.append(np.inf)
        
        # Sort states by volatility: lowest vol state gets index 0
        self.state_map = np.argsort(state_means)
