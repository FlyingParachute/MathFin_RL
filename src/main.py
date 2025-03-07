import pandas as pd
import matplotlib.pyplot as plt
import os
from visualization import plot_fig4_5, plot_fig6_7

def load_ptf1_annual():
    """
    读取 Portfolio_1.xlsx 中的年度数据，并将 'Dates' 列设置为索引.
    """
    file_path = './data/processed/Portfolio_1.xlsx'
    sheets = ['Quarterly', 'Semi Annually', 'Yearly']
    ptf1 = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
    return ptf1['Yearly'].set_index('Dates')

def ensure_dir_exists(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # 读取数据
    ptf1_annual = load_ptf1_annual()
    
    # 确保图片保存目录存在
    figures_dir = './results'
    ensure_dir_exists(figures_dir)
    
    # 绘制论文中的图表
    # Fig.4 & Fig.5: 训练区间 [1976, 2001]，测试区间 (2001, 2016]
    fig4, fig5 = plot_fig4_5(
        data=ptf1_annual,
        train_start_str='1976-01-01',
        train_end_str='2001-12-31',
        test_start_str='2002-01-01',
        test_end_str='2016-12-31',
        freq='annual',
        portfolio='ptf1',
        fig_title_prefix=''
    )
    
    # 保存 Fig.4 和 Fig.5
    fig4.savefig(os.path.join(figures_dir, 'portfolio_performance_on_policy_period1.png'), dpi=300, bbox_inches='tight')
    fig5.savefig(os.path.join(figures_dir, 'portfolio_performance_off_policy_period1.png'), dpi=300, bbox_inches='tight')
    
    # Fig.6 & Fig.7: 训练区间 [1976, 2000]，测试区间 (2000, 2016]
    fig6, fig7 = plot_fig6_7(
        data=ptf1_annual,
        train_start_str='1976-01-01',
        train_end_str='2000-12-31',
        test_start_str='2001-01-01',
        test_end_str='2016-12-31',
        freq='annual',
        portfolio='ptf1',
        fig_title_prefix=''
    )
    
    # 保存 Fig.6 和 Fig.7
    fig6.savefig(os.path.join(figures_dir, 'portfolio_performance_on_policy_period2.png'), dpi=300, bbox_inches='tight')
    fig7.savefig(os.path.join(figures_dir, 'portfolio_performance_off_policy_period2.png'), dpi=300, bbox_inches='tight')
    
    # 显示所有图表
    plt.show()
