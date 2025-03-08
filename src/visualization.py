import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import column_names
from backtest import backtest, backtest_AKA

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
    # 移除对数刻度
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Portfolio Performance: On-policy & Continuous Agents')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Year')
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
    # 移除对数刻度
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Portfolio Performance: Off-policy Agents')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Year')
    plt.tight_layout()
    
    return fig4, fig5

def plot_fig6_7(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq='annual', portfolio='ptf1', fig_title_prefix=''):
    """
    Fig.6: On-policy & Continuous (第二个训练/测试区间)
    Fig.7: Off-policy (第二个训练/测试区间)
    """
    # 与fig4_5同理，只是换了一组训练/测试区间
    fig6, fig7 = plot_fig4_5(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq, portfolio, fig_title_prefix='Portfolio Performance: Second Period - ')
    
    # 重命名图表
    plt.figure(fig6.number)
    plt.title(f'{fig_title_prefix}Portfolio Performance: On-policy & Continuous Agents (Second Period)')
    
    plt.figure(fig7.number)
    plt.title(f'{fig_title_prefix}Portfolio Performance: Off-policy Agents (Second Period)')
    
    return fig6, fig7


# ================
#  绘制 Fig.8 & Fig.9
# ================
def plot_fig8_9(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq='annual', portfolio='ptf3', fig_title_prefix=''):
    """
    Fig.8: On-policy(SARSA) & Continuous
    Fig.9: Off-policy(Q-learning)
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

    # ========== Fig.8: on-policy & continuous ==========
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

    fig8 = plt.figure(figsize=(10, 6))
    plt.plot(ska_dates, ska_vals,   label='SKA (R-SKA)', linewidth=2)
    plt.plot(aka_dates, aka_vals,   label='AKA (R-AKA)', linewidth=2)
    plt.plot(ska_dates, s_ska_vals, label='S-SKA',       linewidth=2)
    plt.plot(ska_dates, s_aka_vals, label='S-AKA',       linewidth=2)
    plt.plot(ska_dates, ca_ska_vals,label='CA-SKA',      linewidth=2)
    plt.plot(ska_dates, ca_aka_vals,label='CA-AKA',      linewidth=2)
    plt.plot(ska_dates, benchmarks['Bonds'],  label='T-NOTE', linestyle='--')
    plt.plot(ska_dates, benchmarks['Stocks'], label='S&P 500',   linestyle='--')
    plt.plot(ska_dates, benchmarks['Ceiling'],label='Ceiling',   linestyle='-.')
    # 移除对数刻度
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Portfolio Performance: On-policy & Continuous Agents')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Year')
    plt.tight_layout()

    # ========== Fig.9: off-policy ==========
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

    fig9 = plt.figure(figsize=(10, 6))
    plt.plot(q_ska_dates, benchmarks['Bonds'],  label='T-NOTE',  linestyle='--')
    plt.plot(q_ska_dates, benchmarks['Stocks'], label='Stocks', linestyle='--')
    plt.plot(q_ska_dates, q_ska_vals,   label='Q-SKA',   linewidth=2)
    plt.plot(q_ska_dates, q_aka_vals,   label='Q-AKA',   linewidth=2)
    plt.plot(q_ska_dates, qs_ska_vals,  label='QS-SKA',  linewidth=2)
    plt.plot(q_ska_dates, qs_aka_vals,  label='QS-AKA',  linewidth=2)
    plt.plot(q_ska_dates, benchmarks['A2'], label='A2', linestyle=':')
    plt.plot(q_ska_dates, benchmarks['A3'], label='A3', linestyle=':')
    plt.plot(q_ska_dates, benchmarks['A4'], label='A4', linestyle=':')
    # 移除对数刻度
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.title(f'{fig_title_prefix}Portfolio Performance: Off-policy Agents')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Year')
    plt.tight_layout()
    
    return fig8, fig9

def plot_fig10_11(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq='annual', portfolio='ptf3', fig_title_prefix=''):
    """
    Fig.10: On-policy & Continuous (第二个训练/测试区间)
    Fig.11: Off-policy (第二个训练/测试区间)
    """
    # 与fig8_9同理，只是换了一组训练/测试区间
    fig10, fig11 = plot_fig8_9(data, train_start_str, train_end_str, test_start_str, test_end_str,
                freq, portfolio, fig_title_prefix='Portfolio Performance: Second Period - ')
    
    # 重命名图表
    plt.figure(fig10.number)
    plt.title(f'{fig_title_prefix}Portfolio Performance: On-policy & Continuous Agents (Second Period)')
    
    plt.figure(fig11.number)
    plt.title(f'{fig_title_prefix}Portfolio Performance: Off-policy Agents (Second Period)')
    
    return fig10, fig11
