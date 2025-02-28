import pandas as pd
import matplotlib.pyplot as plt
from backtest import backtest, backtest_AKA
from visualization import plot_fig4_5, plot_fig6_7

def load_ptf1_annual():
    """
    读取 Portfolio_1.xlsx 中的年度数据，并将 'Dates' 列设置为索引.
    """
    file_path = './data/processed/Portfolio_1.xlsx'
    sheets = ['Quarterly', 'Semi Annually', 'Yearly']
    ptf1 = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
    return ptf1['Yearly'].set_index('Dates')

if __name__ == "__main__":
    # 读取数据
    ptf1_annual = load_ptf1_annual()
    
    # 设置训练和测试区间（示例中采用2000年底为训练结束，2016年底为测试结束）
    train_end = pd.to_datetime('2000-12-31')
    test_end = pd.to_datetime('2016-12-31')
    
    # 运行静态知识代理回测 (Static Knowledge Agent, SKA)
    static_values, static_dates = backtest(
        data=ptf1_annual,
        train_end_date=train_end,
        test_end_date=test_end,
        agent_type='sarsa',
        reward_type='return',
        freq='annual',
        portfolio='ptf1'
    )
    print("Static (SKA) final portfolio value:", static_values[-1])
    
    # 运行自适应知识代理回测 (Adaptive Knowledge Agent, AKA)
    adaptive_values, adaptive_dates = backtest_AKA(
        data=ptf1_annual,
        train_end_date=train_end,
        test_end_date=test_end,
        agent_type='sarsa',
        reward_type='return',
        freq='annual',
        portfolio='ptf1'
    )
    print("Adaptive (AKA) final portfolio value:", adaptive_values[-1])
    
    # 绘制论文中的图表
    # Fig.4 & Fig.5: 训练区间 [1976, 2001]，测试区间 (2001, 2016]
    plot_fig4_5(
        data=ptf1_annual,
        train_start_str='1976-01-01',
        train_end_str='2001-12-31',
        test_start_str='2002-01-01',
        test_end_str='2016-12-31',
        freq='annual',
        portfolio='ptf1',
        fig_title_prefix=''  # 无额外前缀，标题直接为 Fig.4 和 Fig.5
    )
    
    # Fig.6 & Fig.7: 训练区间 [1976, 2000]，测试区间 (2000, 2016]
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
