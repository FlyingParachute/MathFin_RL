import numpy as np
import pandas as pd
from agents import SarsaLambdaAgent, QLambdaAgent, TDContinuousAgent, get_state
from config import column_names

def backtest(data, train_end_date, test_end_date,
             agent_type='sarsa', reward_type='return',
             freq='annual', portfolio='ptf1'):
    """
    Static Knowledge Agents (SKAs).
    """
    spx_col = column_names[portfolio][freq]['spx']
    agg_col = column_names[portfolio][freq]['agg']

    if isinstance(train_end_date, str):
        train_end_date = pd.to_datetime(train_end_date)
    if isinstance(test_end_date, str):
        test_end_date = pd.to_datetime(test_end_date)

    train_data = data[data.index <= train_end_date]
    test_data = data[(data.index > train_end_date) & (data.index <= test_end_date)]

    # Initialize agent
    if agent_type == 'continuous':
        agent = TDContinuousAgent()
    elif agent_type == 'sarsa':
        agent = SarsaLambdaAgent(reward_type=reward_type)
    elif agent_type == 'qlearning':
        agent = QLambdaAgent(reward_type=reward_type)

    # Training phase: multiple episodes with randomized initial states
    num_episodes = 50
    min_ep_len = 4
    for _ in range(num_episodes):
        # Randomly select starting point
        start_idx = np.random.randint(0, max(1, len(train_data) - min_ep_len))
        
        # Reset eligibility traces
        if agent_type == 'continuous':
            agent.e_trace = {s: [0, 0] for s in agent.e_trace}
        else:
            agent.e = pd.DataFrame(np.zeros((4, 5)), index=agent.q.index, columns=agent.q.columns)
            
        for i in range(start_idx + 1, len(train_data)):
            prev_row = train_data.iloc[i - 1]
            curr_row = train_data.iloc[i]
            state = get_state(prev_row[spx_col], prev_row[agg_col])

            if agent_type == 'continuous':
                action = agent.get_allocation(state)
            else:
                action = agent.choose_action(state)

            # Calculate reward
            if agent_type in ['sarsa', 'qlearning'] and reward_type == 'sharpe':
                reward = agent.get_reward(curr_row[spx_col], curr_row[agg_col], action)
            else:
                portfolio_ret = action * curr_row[spx_col] + (1 - action) * curr_row[agg_col]
                reward = portfolio_ret

            next_state = get_state(curr_row[spx_col], curr_row[agg_col])

            if agent_type == 'sarsa':
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
            elif agent_type == 'qlearning':
                agent.update(state, action, reward, next_state)
            elif agent_type == 'continuous':
                agent.update(state, curr_row[spx_col], curr_row[agg_col], reward, next_state)

    # Testing phase
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
    Adaptive Knowledge Agents (AKAs).
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

        # Reinitialize agent each time
        if agent_type == 'continuous':
            agent = TDContinuousAgent()
        elif agent_type == 'sarsa':
            agent = SarsaLambdaAgent(reward_type=reward_type)
        elif agent_type == 'qlearning':
            agent = QLambdaAgent(reward_type=reward_type)

        # Training
        num_episodes = 50  # Increase number of iterations
        min_ep_len = 4
        if len(current_train_data) > min_ep_len:
            for _ in range(num_episodes):
                start_idx = np.random.randint(0, len(current_train_data) - min_ep_len)
                
                # Reset eligibility traces
                if agent_type == 'continuous':
                    agent.e_trace = {s: [0, 0] for s in agent.e_trace}
                else:
                    agent.e = pd.DataFrame(np.zeros((4, 5)), index=agent.q.index, columns=agent.q.columns)
                    
                for i in range(start_idx + 1, len(current_train_data)):
                    prev_row = current_train_data.iloc[i - 1]
                    curr_row = current_train_data.iloc[i]
                    state = get_state(prev_row[spx_col], prev_row[agg_col])

                    if agent_type == 'continuous':
                        action = agent.get_allocation(state)
                    else:
                        action = agent.choose_action(state)

                    if agent_type in ['sarsa', 'qlearning'] and reward_type == 'sharpe':
                        reward = agent.get_reward(curr_row[spx_col], curr_row[agg_col], action)
                    else:
                        portfolio_ret = action * curr_row[spx_col] + (1 - action) * curr_row[agg_col]
                        reward = portfolio_ret

                    next_state = get_state(curr_row[spx_col], curr_row[agg_col])

                    if agent_type == 'sarsa':
                        next_action = agent.choose_action(next_state)
                        agent.update(state, action, reward, next_state, next_action)
                    elif agent_type == 'qlearning':
                        agent.update(state, action, reward, next_state)
                    elif agent_type == 'continuous':
                        agent.update(state, curr_row[spx_col], curr_row[agg_col], reward, next_state)

        # Test current point
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
