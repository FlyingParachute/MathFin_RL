import numpy as np
import pandas as pd
from config import gamma, lambda_, alpha, epsilon, eta

def get_state(spx_ret, agg_ret):
    """将spx和agg的正负收益转换为二进制状态."""
    s = ''
    s += '1' if spx_ret >= 0 else '0'
    s += '1' if agg_ret >= 0 else '0'
    return s

# 离散动作空间(股票比例)
actions = [0.0, 0.25, 0.5, 0.75, 1.0]

def initialize_q():
    """初始化Q矩阵(4状态x5动作)."""
    states = ['11', '01', '10', '00']
    return pd.DataFrame(np.random.rand(4, 5), index=states, columns=actions)

class SarsaLambdaAgent:
    """
    SARSA(λ)离散动作代理.
    reward_type='return' 或 'sharpe'.
    """
    def __init__(self, reward_type='return'):
        self.q = initialize_q()
        self.e = pd.DataFrame(np.zeros((4, 5)), index=self.q.index, columns=self.q.columns)
        self.reward_type = reward_type
        self.A = 0  # 用于差分夏普比率的第一阶矩
        self.B = 0  # 用于差分夏普比率的第二阶矩

    def get_reward(self, spx_ret, agg_ret, action):
        portfolio_ret = action * spx_ret + (1 - action) * agg_ret
        if self.reward_type == 'return':
            return portfolio_ret
        else:
            # 使用旧的A、B计算差分夏普比率
            old_A = self.A
            old_B = self.B
            denominator = (old_B - old_A**2)**1.5
            if abs(denominator) < 1e-6:
                dsr = 0
            else:
                dsr = (old_B * (portfolio_ret - old_A) - 0.5 * old_A * (portfolio_ret**2 - old_B)) / denominator
            # 更新A、B
            self.A = old_A + eta * (portfolio_ret - old_A)
            self.B = old_B + eta * (portfolio_ret**2 - old_B)
            return dsr

    def update(self, state, action, reward, next_state, next_action):
        delta = reward + gamma * self.q.loc[next_state, next_action] - self.q.loc[state, action]
        # 替换迹更新
        self.e = self.e * gamma * lambda_
        self.e.loc[state, action] = 1
        self.q += alpha * delta * self.e

    def choose_action(self, state):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            return self.q.loc[state].idxmax()

class QLambdaAgent(SarsaLambdaAgent):
    """
    Q(λ)离散动作代理.
    reward_type='return' 或 'sharpe'.
    """
    def __init__(self, reward_type='return'):
        super().__init__(reward_type)

    def update(self, state, action, reward, next_state):
        delta = reward + gamma * self.q.loc[next_state].max() - self.q.loc[state, action]
        self.e = self.e * gamma * lambda_
        self.e.loc[state, action] = 1
        self.q += alpha * delta * self.e

class TDContinuousAgent:
    """
    TD(λ)连续动作代理(两资产).
    """
    def __init__(self):
        # 每个状态有theta=[theta1, theta2], theta1 in [0,1]代表股票比例
        self.theta = {
            '11': [0.5, 0], '01': [0.5, 0],
            '10': [0.5, 0], '00': [0.5, 0]
        }
        self.e_trace = {
            '11': [0, 0], '01': [0, 0],
            '10': [0, 0], '00': [0, 0]
        }

    def get_allocation(self, state):
        if np.random.rand() < epsilon:
            return np.random.uniform(0, 1)
        else:
            return np.clip(self.theta[state][0], 0, 1)

    def update(self, state, spx_ret, agg_ret, reward, next_state):
        gradient = [spx_ret - agg_ret, 1]
        current_value = self.theta[state][0] * (spx_ret - agg_ret) + self.theta[state][1]
        next_value = self.theta[next_state][0] * (spx_ret - agg_ret) + self.theta[next_state][1]
        delta = reward + gamma * next_value - current_value
        # 资格迹更新
        for i in range(2):
            self.e_trace[state][i] = gamma * lambda_ * self.e_trace[state][i] + gradient[i]
        # 参数更新
        for i in range(2):
            self.theta[state][i] += alpha * delta * self.e_trace[state][i]
        # 约束theta1在[0,1]
        self.theta[state][0] = np.clip(self.theta[state][0], 0, 1)