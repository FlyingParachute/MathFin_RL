import numpy as np
import pandas as pd
from config import gamma, lambda_, alpha, epsilon, eta

def get_state(spx_ret, agg_ret):
    """Convert positive and negative returns of spx and agg to binary states."""
    s = ''
    s += '1' if spx_ret >= 0 else '0'
    s += '1' if agg_ret >= 0 else '0'
    return s

# Discrete action space (stock proportion)
actions = [0.0, 0.25, 0.5, 0.75, 1.0]

def initialize_q():
    """Initialize Q matrix (4 states x 5 actions)."""
    states = ['11', '01', '10', '00']
    return pd.DataFrame(np.random.rand(4, 5), index=states, columns=actions)

class SarsaLambdaAgent:
    """
    SARSA(λ) discrete action agent.
    reward_type='return' or 'sharpe'.
    """
    def __init__(self, reward_type='return'):
        self.q = initialize_q()
        self.e = pd.DataFrame(np.zeros((4, 5)), index=self.q.index, columns=self.q.columns)
        self.reward_type = reward_type
        self.A = 0  # First moment for differential Sharpe ratio
        self.B = 0  # Second moment for differential Sharpe ratio

    def get_reward(self, spx_ret, agg_ret, action):
        portfolio_ret = action * spx_ret + (1 - action) * agg_ret
        
        if self.reward_type == 'return':
            return portfolio_ret
        else:  # sharpe
            # 使用旧的A、B计算差分夏普比率
            old_A = self.A
            old_B = self.B
            denominator = (old_B - old_A**2)**1.5
            
            if denominator == 0:
                dsr = 0
            else:
                dsr = (old_B * (portfolio_ret - old_A) - 0.5 * old_A * (portfolio_ret**2 - old_B)) / denominator
            
            # 更新A、B
            self.A = old_A + eta * (portfolio_ret - old_A)
            self.B = old_B + eta * (portfolio_ret**2 - old_B)
            return dsr

    def update(self, state, action, reward, next_state, next_action):
        # Calculate TD error
        delta = reward + gamma * self.q.loc[next_state, next_action] - self.q.loc[state, action]
        
        # Replace trace update - set current state-action pair to 1
        self.e.loc[state, action] = 1
        
        # Update Q values
        for s in self.q.index:
            for a in self.q.columns:
                self.q.loc[s, a] += alpha * delta * self.e.loc[s, a]
        
        # Decay all eligibility traces
        self.e = gamma * lambda_ * self.e

    def choose_action(self, state):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            return self.q.loc[state].idxmax()

class QLambdaAgent(SarsaLambdaAgent):
    """
    Q(λ) discrete action agent.
    reward_type='return' or 'sharpe'.
    """
    def __init__(self, reward_type='return'):
        super().__init__(reward_type)

    def update(self, state, action, reward, next_state):
        # Find the action with maximum Q value for the next state
        a_star = self.q.loc[next_state].idxmax()
        
        # Calculate TD error
        delta = reward + gamma * self.q.loc[next_state, a_star] - self.q.loc[state, action]
        
        # Set eligibility trace for the current state-action pair to 1
        self.e.loc[state, action] = 1
        
        # Update Q values
        for s in self.q.index:
            for a in self.q.columns:
                self.q.loc[s, a] += alpha * delta * self.e.loc[s, a]
        
        # Update eligibility traces based on greedy action
        # If the next action is not greedy, zero all eligibility traces
        next_action = self.choose_action(next_state)
        if next_action != a_star:
            self.e = 0 * self.e
        else:
            self.e = gamma * lambda_ * self.e

class TDContinuousAgent:
    """
    TD(λ) continuous action agent (two assets).
    """
    def __init__(self):
        # Each state has theta=[theta1, theta2], theta1 in [0,1] represents stock proportion
        states = ['11', '01', '10', '00']
        self.theta = {s: [np.random.uniform(0, 1), 0] for s in states}
        self.e_trace = {s: [0, 0] for s in states}

    def get_value(self, state, spx_ret, agg_ret):
        """Calculate the value function for the current state."""
        # V(s) = θ₁ᴱ(R_t^S - R_t^B) + θ₂ᴱ
        return self.theta[state][0] * (spx_ret - agg_ret) + self.theta[state][1]

    def get_allocation(self, state):
        """Return stock allocation proportion based on current state."""
        if np.random.rand() < epsilon:
            # Exploration: return a random value between [0,1]
            return np.random.uniform(0, 1)
        else:
            # Exploitation: return the θ₁ value for the current state
            return np.clip(self.theta[state][0], 0, 1)

    def update(self, state, spx_ret, agg_ret, reward, next_state):
        # Calculate value functions for current and next states
        current_value = self.get_value(state, spx_ret, agg_ret)
        next_value = self.get_value(next_state, spx_ret, agg_ret)
        
        # Calculate TD error
        delta = reward + gamma * next_value - current_value
        
        # Update eligibility trace: e = γλe + ∇θV(s)
        gradient = [spx_ret - agg_ret, 1]  # ∇θV(s) = (R_t^S - R_t^B, 1)^T
        for i in range(2):
            self.e_trace[state][i] = gamma * lambda_ * self.e_trace[state][i] + gradient[i]
        
        # Update parameters: θ = θ + αδe
        for i in range(2):
            self.theta[state][i] += alpha * delta * self.e_trace[state][i]
        
        # Constrain θ₁ to the [0,1] interval
        self.theta[state][0] = np.clip(self.theta[state][0], 0, 1)
