import numpy as np

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        # We start with a Beta(1, 1) prior for each arm, which is uniform
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.n_arms = n_arms
        
    def select_arm(self):
        # Sample a value from the Beta distribution for each arm
        theta_samples = np.random.beta(self.alpha, self.beta)
        # Return the arm with the highest sampled value
        return np.argmax(theta_samples)
    
    def update(self, chosen_arm, reward):
        # Update the Beta distribution parameters for the chosen arm
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        # Track the number of times each arm has been pulled
        self.counts = np.zeros(n_arms)
        # Track the total rewards for each arm
        self.values = np.zeros(n_arms)
        
    def select_arm(self):
        # With probability epsilon, explore (choose randomly)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        # Otherwise, exploit (choose the best known arm)
        else:
            # For arms that haven't been tried, treat them as having value 0
            current_values = np.divide(self.values, self.counts, 
                                     out=np.zeros_like(self.values), 
                                     where=self.counts != 0)
            return np.argmax(current_values)
    
    def update(self, chosen_arm, reward):
        # Update the count for this arm
        self.counts[chosen_arm] += 1
        # Update the value using incremental average
        # value = (value * (count-1) + reward) / count
        self.values[chosen_arm] += (reward - self.values[chosen_arm] / self.counts[chosen_arm])

class UCBandit:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        # Track the number of times each arm has been pulled
        self.counts = np.zeros(n_arms)
        # Track the total rewards for each arm
        self.values = np.zeros(n_arms)
        self.total_rounds = 0
        
    def select_arm(self):
        self.total_rounds += 1
        
        # For arms that haven't been tried yet, try them first
        untried_arms = np.where(self.counts == 0)[0]
        if len(untried_arms) > 0:
            return untried_arms[0]
        
        # Calculate UCB values for each arm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            exploitation = self.values[arm] / self.counts[arm]
            exploration = self.c * np.sqrt(np.log(self.total_rounds) / self.counts[arm])
            ucb_values[arm] = exploitation + exploration
            
        return np.argmax(ucb_values)
    
    def update(self, chosen_arm, reward):
        # Update the count for this arm
        self.counts[chosen_arm] += 1
        # Update the value using incremental average
        # value = (value * (count-1) + reward) / count
        self.values[chosen_arm] += (reward - self.values[chosen_arm] / self.counts[chosen_arm])