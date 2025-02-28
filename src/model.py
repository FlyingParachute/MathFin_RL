import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

# ==============================
# 1. 数据读取与全局配置
# ==============================
file_path_1 = './data/processed/Portfolio_1.xlsx'
sheets = ['Quarterly', 'Semi Annually', 'Yearly']
ptf1 = {sheet: pd.read_excel(file_path_1, sheet_name=sheet) for sheet in sheets}

ptf1_quarterly = ptf1['Quarterly'].set_index('Dates')
ptf1_semi_annual = ptf1['Semi Annually'].set_index('Dates')
ptf1_annual = ptf1['Yearly'].set_index('Dates')

file_path_2 = './data/processed/Portfolio_2.xlsx'
sheets = ['Quarterly', 'Semi Annually', 'Yearly']
ptf2 = {sheet: pd.read_excel(file_path_2, sheet_name=sheet) for sheet in sheets}

ptf2_quarterly = ptf2['Quarterly'].set_index('Dates')
ptf2_semi_annual = ptf2['Semi Annually'].set_index('Dates')
ptf2_annual = ptf2['Yearly'].set_index('Dates')

# 不同频率下的列名映射
column_names = {
    'ptf1': {
        'quarterly':  {'spx': 'SPXret_1q', 'agg': 'AGGret_1q'},
        'semi_annual':{'spx': 'SPXret_s',  'agg': 'AGGret_s'},
        'annual':     {'spx': 'SPXret_a',  'agg': 'AGGret_a'}
    },
    'ptf2': {
        'quarterly':  {'spx': 'SPXret_1q', 'agg': 'TNXret_1q'},
        'semi_annual':{'spx': 'SPXret_s',  'agg': 'TNXret_s'},
        'annual':     {'spx': 'SPXret_a',  'agg': 'TNXret_a'}
    }
}

# 全局参数
gamma   = 0.9     # 折扣因子
lambda_ = 0.9     # 资格迹衰减率
alpha   = 0.1     # 学习率
epsilon = 0.01    # 探索概率
eta     = 0.1     # 差分夏普比率平滑参数

# ==============================
# 2. 工具函数和状态编码
# ==============================
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

# ==============================
# 3. 不同RL算法类定义
# ==============================
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

# ==============================
# 4. 回测框架（静态SKA & 自适应AKA）
# ==============================
def backtest(data, train_end_date, test_end_date,
             agent_type='sarsa', reward_type='return',
             freq='annual', portfolio='ptf1'):
    """
    静态知识代理(SKAs).
    """
    spx_col = column_names[portfolio][freq]['spx']
    agg_col = column_names[portfolio][freq]['agg']

    if isinstance(train_end_date, str):
        train_end_date = pd.to_datetime(train_end_date)
    if isinstance(test_end_date, str):
        test_end_date = pd.to_datetime(test_end_date)

    train_data = data[data.index <= train_end_date]
    test_data  = data[(data.index > train_end_date) & (data.index <= test_end_date)]

    # 初始化代理
    if agent_type == 'continuous':
        agent = TDContinuousAgent()
    elif agent_type == 'sarsa':
        agent = SarsaLambdaAgent(reward_type=reward_type)
    elif agent_type == 'qlearning':
        agent = QLambdaAgent(reward_type=reward_type)

    # 训练阶段：多episode随机化初始状态
    num_episodes = 10
    min_ep_len = 4
    for _ in range(num_episodes):
        start_idx = np.random.randint(0, max(1, len(train_data) - min_ep_len))
        for i in range(start_idx + 1, len(train_data)):
            prev_row = train_data.iloc[i - 1]
            curr_row = train_data.iloc[i]
            state = get_state(prev_row[spx_col], prev_row[agg_col])

            if agent_type == 'continuous':
                action = agent.get_allocation(state)
            else:
                action = agent.choose_action(state)

            # 计算reward
            if (agent_type in ['sarsa', 'qlearning'] and reward_type == 'sharpe'):
                reward = agent.get_reward(curr_row[spx_col], curr_row[agg_col], action)
            else:
                reward = action * curr_row[spx_col] + (1 - action) * curr_row[agg_col]

            next_state = get_state(curr_row[spx_col], curr_row[agg_col])

            if agent_type == 'sarsa':
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
            elif agent_type == 'qlearning':
                agent.update(state, action, reward, next_state)
            elif agent_type == 'continuous':
                agent.update(state, curr_row[spx_col], curr_row[agg_col], reward, next_state)

    # 测试阶段
    portfolio_values = [10000]
    current_value = 10000
    dates = [train_end_date]

    for i in range(len(test_data)):
        if i == 0:
            prev_row = train_data.iloc[-1]
        else:
            prev_row = test_data.iloc[i - 1]
        curr_row = test_data.iloc[i]

        state = get_state(prev_row[spx_col], prev_row[agg_col])
        if agent_type == 'continuous':
            action = agent.get_allocation(state)
        else:
            action = agent.choose_action(state)

        ret = action * curr_row[spx_col] + (1 - action) * curr_row[agg_col]
        current_value *= (1 + ret)
        portfolio_values.append(current_value)
        dates.append(test_data.index[i])

    return portfolio_values, dates

def backtest_AKA(data, train_end_date, test_end_date,
                 agent_type='sarsa', reward_type='return',
                 freq='annual', portfolio='ptf1'):
    """
    自适应知识代理(AKAs).
    """
    spx_col = column_names[portfolio][freq]['spx']
    agg_col = column_names[portfolio][freq]['agg']

    full_data = pd.concat([
        data[data.index <= train_end_date],
        data[(data.index > train_end_date) & (data.index <= test_end_date)]
    ])

    portfolio_values = [10000]
    dates = [train_end_date]

    for current_date in full_data[full_data.index > train_end_date].index:
        current_train_data = full_data[full_data.index < current_date]

        # 每次都重新初始化
        if agent_type == 'continuous':
            agent = TDContinuousAgent()
        elif agent_type == 'sarsa':
            agent = SarsaLambdaAgent(reward_type=reward_type)
        elif agent_type == 'qlearning':
            agent = QLambdaAgent(reward_type=reward_type)

        # 训练
        num_episodes = 10
        min_ep_len = 4
        if len(current_train_data) > min_ep_len:
            for _ in range(num_episodes):
                start_idx = np.random.randint(0, len(current_train_data) - min_ep_len)
                for i in range(start_idx + 1, len(current_train_data)):
                    prev_row = current_train_data.iloc[i - 1]
                    curr_row = current_train_data.iloc[i]
                    state = get_state(prev_row[spx_col], prev_row[agg_col])

                    if agent_type == 'continuous':
                        action = agent.get_allocation(state)
                    else:
                        action = agent.choose_action(state)

                    if (agent_type in ['sarsa', 'qlearning'] and reward_type == 'sharpe'):
                        reward = agent.get_reward(curr_row[spx_col], curr_row[agg_col], action)
                    else:
                        reward = action * curr_row[spx_col] + (1 - action) * curr_row[agg_col]

                    next_state = get_state(curr_row[spx_col], curr_row[agg_col])

                    if agent_type == 'sarsa':
                        next_action = agent.choose_action(next_state)
                        agent.update(state, action, reward, next_state, next_action)
                    elif agent_type == 'qlearning':
                        agent.update(state, action, reward, next_state)
                    elif agent_type == 'continuous':
                        agent.update(state, curr_row[spx_col], curr_row[agg_col], reward, next_state)

        # 测试当前点
        prev_row = current_train_data.iloc[-1] if len(current_train_data) > 0 else full_data.iloc[0]
        current_row = full_data.loc[current_date]
        state = get_state(prev_row[spx_col], prev_row[agg_col])

        if agent_type == 'continuous':
            action = agent.get_allocation(state)
        else:
            action = agent.choose_action(state)

        ret = action * current_row[spx_col] + (1 - action) * current_row[agg_col]
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        dates.append(current_date)

    return portfolio_values, dates

# ==============================
# 5. 基准策略 & 可视化
# ==============================
def calculate_benchmarks(data, train_end_date, test_end_date,
                         freq='annual', portfolio='ptf1'):
    spx_col = column_names[portfolio][freq]['spx']
    agg_col = column_names[portfolio][freq]['agg']

    test_data = data[(data.index > train_end_date) & (data.index <= test_end_date)]

    allocations = {
        'A2': 0.25,
        'A3': 0.5,
        'A4': 0.75,
        'Bonds': 0.0,
        'Stocks': 1.0
    }
    # 天花板策略
    benchmarks = {k: [10000] for k in allocations}
    benchmarks['Ceiling'] = [10000]

    for i in range(len(test_data)):
        spx_ret = test_data.iloc[i][spx_col]
        agg_ret = test_data.iloc[i][agg_col]
        for strategy, alloc in allocations.items():
            ret = alloc * spx_ret + (1 - alloc) * agg_ret
            benchmarks[strategy].append(benchmarks[strategy][-1] * (1 + ret))
        best_ret = max(spx_ret, agg_ret)
        benchmarks['Ceiling'].append(benchmarks['Ceiling'][-1] * (1 + best_ret))
    return benchmarks

def annualized_returns(final_val, years):
    """
    final_val: 期末投资组合价值
    years:     投资年数
    返回总收益和年化收益
    """
    total_ret = (final_val / 10000 - 1) * 100
    ann_ret   = ((1 + total_ret/100)**(1/years) - 1) * 100
    return total_ret, ann_ret

# ================
#  绘制 Fig.4 & Fig.5
# ================
def plot_fig4_5(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq='annual', portfolio='ptf1', fig_title_prefix=''):
    """
    Fig.4: On-policy(SARSA) & Continuous
    Fig.5: Off-policy(Q-learning)
    训练区间: [train_start, train_end]
    测试区间: (train_end, test_end]
    """
    train_start = pd.to_datetime(train_start_str)
    train_end   = pd.to_datetime(train_end_str)
    test_start  = pd.to_datetime(test_start_str)
    test_end    = pd.to_datetime(test_end_str)

    # 取出这段区间的数据
    full_data = data[(data.index >= train_start) & (data.index <= test_end)]
    # 在下面的 backtest 中会再次筛选
    # 这里只做保证数据不越界

    # 基准
    benchmarks = calculate_benchmarks(full_data, train_end, test_end, freq, portfolio)

    # ========== Fig.4: on-policy & continuous ==========
    # SKA(R-SKA)
    ska_vals, ska_dates = backtest(full_data, train_end, test_end,
                                   agent_type='sarsa', reward_type='return',
                                   freq=freq, portfolio=portfolio)
    # AKA(R-AKA)
    aka_vals, aka_dates = backtest_AKA(full_data, train_end, test_end,
                                       agent_type='sarsa', reward_type='return',
                                       freq=freq, portfolio=portfolio)
    # S-SKA
    s_ska_vals, _ = backtest(full_data, train_end, test_end,
                             agent_type='sarsa', reward_type='sharpe',
                             freq=freq, portfolio=portfolio)
    # S-AKA
    s_aka_vals, _ = backtest_AKA(full_data, train_end, test_end,
                                 agent_type='sarsa', reward_type='sharpe',
                                 freq=freq, portfolio=portfolio)
    # CA-SKA
    ca_ska_vals, _ = backtest(full_data, train_end, test_end,
                              agent_type='continuous', reward_type='return',
                              freq=freq, portfolio=portfolio)
    # CA-AKA
    ca_aka_vals, _ = backtest_AKA(full_data, train_end, test_end,
                                  agent_type='continuous', reward_type='return',
                                  freq=freq, portfolio=portfolio)

    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(ska_dates, ska_vals,   label='SKA (R-SKA)', linewidth=2)
    plt.plot(aka_dates, aka_vals,   label='AKA (R-AKA)', linewidth=2)
    plt.plot(ska_dates, s_ska_vals, label='S-SKA',       linewidth=2)
    plt.plot(ska_dates, s_aka_vals, label='S-AKA',       linewidth=2)
    plt.plot(ska_dates, ca_ska_vals,label='CA-SKA',      linewidth=2)
    plt.plot(ska_dates, ca_aka_vals,label='CA-AKA',      linewidth=2)
    plt.plot(ska_dates, benchmarks['Bonds'],  label='AGG Bonds', linestyle='--')
    plt.plot(ska_dates, benchmarks['Stocks'], label='S&P 500',   linestyle='--')
    plt.plot(ska_dates, benchmarks['Ceiling'],label='Ceiling',   linestyle='-.')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Fig.4 - On-policy & Continuous')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # ========== Fig.5: off-policy ==========
    # Q-SKA
    q_ska_vals, q_ska_dates = backtest(full_data, train_end, test_end,
                                       agent_type='qlearning', reward_type='return',
                                       freq=freq, portfolio=portfolio)
    # Q-AKA
    q_aka_vals, _ = backtest_AKA(full_data, train_end, test_end,
                                 agent_type='qlearning', reward_type='return',
                                 freq=freq, portfolio=portfolio)
    # QS-SKA
    qs_ska_vals, _ = backtest(full_data, train_end, test_end,
                              agent_type='qlearning', reward_type='sharpe',
                              freq=freq, portfolio=portfolio)
    # QS-AKA
    qs_aka_vals, _ = backtest_AKA(full_data, train_end, test_end,
                                  agent_type='qlearning', reward_type='sharpe',
                                  freq=freq, portfolio=portfolio)

    fig5 = plt.figure(figsize=(10, 6))
    plt.plot(q_ska_dates, benchmarks['Bonds'],  label='Bonds',  linestyle='--')
    plt.plot(q_ska_dates, benchmarks['Stocks'], label='Stocks', linestyle='--')
    plt.plot(q_ska_dates, q_ska_vals,   label='Q-SKA',   linewidth=2)
    plt.plot(q_ska_dates, q_aka_vals,   label='Q-AKA',   linewidth=2)
    plt.plot(q_ska_dates, qs_ska_vals,  label='QS-SKA',  linewidth=2)
    plt.plot(q_ska_dates, qs_aka_vals,  label='QS-AKA',  linewidth=2)
    plt.plot(q_ska_dates, benchmarks['A2'], label='A2', linestyle=':')
    plt.plot(q_ska_dates, benchmarks['A3'], label='A3', linestyle=':')
    plt.plot(q_ska_dates, benchmarks['A4'], label='A4', linestyle=':')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Fig.5 - Off-policy')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

def plot_fig6_7(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq='annual', portfolio='ptf1', fig_title_prefix=''):
    """
    Fig.6: On-policy & Continuous (第二个训练/测试区间)
    Fig.7: Off-policy (第二个训练/测试区间)
    """
    # 与fig4_5同理，只是换了一组训练/测试区间
    plot_fig4_5(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq, portfolio, fig_title_prefix='(Second) ' + fig_title_prefix)

# ==============================
# 6. 主函数入口
# ==============================
if __name__ == "__main__":
    # 示例：以PTF1的年度数据进行实验
    # 论文中的 Fig.4 & Fig.5：训练区间[1976,2001], 测试区间(2001,2016]
    # 论文中的 Fig.6 & Fig.7：训练区间[1976,2000], 测试区间(2000,2016]
    #
    # 此处仅演示，具体年份可按需求调整或用实际数据年份。
    
    print("===== Generating Figures 4 & 5 =====")
    plot_fig4_5(
        data=ptf1_annual,
        train_start_str='1976-01-01',
        train_end_str='2001-12-31',
        test_start_str='2002-01-01',
        test_end_str='2016-12-31',
        freq='annual',
        portfolio='ptf1',
        fig_title_prefix=''
    )
    
    print("===== Generating Figures 6 & 7 =====")
    plot_fig6_7(
        data=ptf1_annual,
        train_start_str='1976-01-01',
        train_end_str='2000-12-31',
        test_start_str='2001-01-01',
        test_end_str='2016-12-31',
        freq='annual',
        portfolio='ptf1',
        fig_title_prefix=''
    )
    
    plt.show()
