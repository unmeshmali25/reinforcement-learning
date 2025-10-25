# Setup 4 x 4 grid
import numpy as np

class GridWorld: 
    def __init__(self, size=4): 
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.actions = ['U', 'D', 'L', 'R']
        self.state = self.start

    def reset(self): 
        self.state = self.start
        return self.state
    
    def step(self, action): 
        x,y =  self.state
        if action == 'U': 
            x = max(x-1,0)
        elif action == 'D': 
            x = min(x+1, self.size-1)
        elif action == 'L': 
            y = max(y-1,0)
        elif action == 'R': 
            y = min(y+1, self.size-1)
        
        self.state = (x,y)
        reward = -1 # small penalty for each step
        if self.state == self.goal: 
            reward = 10
            done = True
        else: 
            done = False
        
        return self.state, reward, done
    
    def get_all_states(self): 
        return [(x,y) for x in range(self.size) for y in range(self.size)]

    def is_terminal(self, state): 
        return state == self.goal


class GridWorldObstacle: 
    def __init__(self, size=4, obstacle=[(2,3)]): 
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.actions = ['U', 'D', 'L', 'R']
        self.state = self.start
        self.obstacle = obstacle

    def reset(self): 
        self.state = self.start
        return self.state
    
    def step(self, action): 
        x,y =  self.state
        if action == 'U': 
            x = max(x-1,0)
        elif action == 'D': 
            x = min(x+1, self.size-1)
        elif action == 'L': 
            y = max(y-1,0)
        elif action == 'R': 
            y = min(y+1, self.size-1)
        
        if (x,y) in self.obstacle:
            reward = -1000
            done = False # hitting an obstacle doesn't end the episode
        else:
            self.state = (x,y)
            reward = -1 # default step penalty
    
            if self.state == self.goal: 
                reward = 10
                done = True
            else: 
                done = False
        
        return self.state, reward, done
    
    def get_all_states(self): 
        # I don't want to return states that are obstacles
        all_states = [(x,y) for x in range(self.size) for y in range(self.size)]
        return list(set(all_states) - set(self.obstacle))

    def is_terminal(self, state): 
        return state == self.goal


class GridWorldStochasticActions():
    def __init__(self, size = 4, obstacle = [(2,3)]): 
        self.size = size
        self.grid = np.zeros((size, size))
        self.goal = (size - 1, size - 1) 
        self.actions = ['U', 'D', 'L', 'R']
        self.state = self.start
        self.obstacle = obstacle

    def reset(self):
        self.state = start
        return self.state

    def step(self): 
        x,y = self.state

        # Define transition probabilities
        action_probabilities = {
            'U' : [('U', 0.8), ('L', 0.1), ('R', 1)]
            'D': [('D', 0.5), ('L', 0.1), ('R', 0.4)],
            'L': [('L', 0.8), ('U', 0.1), ('D', 0.1)],
            'R': [('R', 0.8), ('U', 0.1), ('D', 0.1)],
        }

        possible_actions = list(action_probabilities[action].keys())
        probabilities = list(action_probabilities[action].values())

        # actual vs intended action
        actual_action = np.random.choice(possible_actions, p = probabilities)

        #TODO
        if actual_action == 'U': 
            



# Environment factory function
def create_environment(env_type, size=4, obstacle=None):
    """Factory function to create different types of GridWorld environments"""
    if env_type == 'basic':
        return GridWorld(size=size)
    elif env_type == 'obstacle':
        obstacle = obstacle or [(2,3)]
        return GridWorldObstacle(size=size, obstacle=obstacle)
    elif env_type == 'stochastic':
        obstacle = obstacle or [(2,3)]
        return GridWorldStochasticActions(size=size, obstacle=obstacle)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

if __name__ == "__main__":
    env = GridWorld(size=4)
    # env = GridWorldObstacle(size=4, obstacle=[(1,3)])
    




