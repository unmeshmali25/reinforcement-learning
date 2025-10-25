import numpy as np 
import pandas as pd 


def value_iteration(env, gamma=0.9, theta=1e-6): 
    V = {state: 0 for state in env.get_all_states()}
    policy = {}

    while True: 
        delta = 0
        for state in env.get_all_states(): 
            if env.is_terminal(state): 
                continue
            v = V[state]
            action_values = []

            for action in env.actions:
                env.state = state # temporarily set state
                next_state, reward, _ = env.step(action)
                value = reward + gamma * V[next_state]
                action_values.append(value)
            V[state] = max(action_values)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break
    
    # Extract Policy
    for state in env.get_all_states():
        if env.is_terminal(state): 
            policy[state] = None
            continue

        action_values = []
        for action in env.actions:
            env.state = state
            next_state, reward, _ = env.step(action)
            value = reward + gamma * V[next_state]
            action_values.append(value)

        best_action = env.actions[np.argmax(action_values)]
        policy[state] = best_action

    return V, policy


def policy_iteration(env, gamma=0.9, theta=1e-6):
    """Policy iteration algorithm"""
    V = {state: 0 for state in env.get_all_states()}
    policy = {state: env.actions[0] for state in env.get_all_states() 
              if not env.is_terminal(state)}
    
    is_policy_stable = False
    
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for state in env.get_all_states():
                if env.is_terminal(state):
                    continue
                    
                v = V[state]
                action = policy[state]
                env.state = state
                next_state, reward, _ = env.step(action)
                V[state] = reward + gamma * V[next_state]
                delta = max(delta, abs(v - V[state]))
            
            if delta < theta:
                break
        
        # Policy Improvement
        is_policy_stable = True
        for state in env.get_all_states():
            if env.is_terminal(state):
                continue
                
            old_action = policy[state]
            action_values = []
            
            for action in env.actions:
                env.state = state
                next_state, reward, _ = env.step(action)
                value = reward + gamma * V[next_state]
                action_values.append(value)
            
            best_action = env.actions[np.argmax(action_values)]
            policy[state] = best_action
            
            if old_action != best_action:
                is_policy_stable = False
    
    return V, policy


# Policy factory function
def create_policy(policy_type, env, **kwargs):
    """Factory function to create different types of policies"""
    if policy_type == 'value_iteration':
        return value_iteration(env, **kwargs)
    elif policy_type == 'policy_iteration':
        return policy_iteration(env, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


if __name__ == "__main__":
    pass








