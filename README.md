# RL大作业

> **注意：** model.py和model.ipynb都是不完善的版本，我打算把代码分成几个文件，调顺溜了放在model.ipynb里作为提交.

## 数据类
- **`data/processed/Portfolio_1.xlsx`**: SPX和AGG的数据
- **`data/processed/Portfolio_2.xlsx`**: SPX和TNX的数据
- **`src/data_process.ipynb`**: 数据处理的notebook，直接用/data/processed/下数据的话不用管

## 当前版本目录
- **`main.py`**: 主程序
- **`agents.py`**: 文章中三种agent的实现
- **`config.py`**: 配置文件，现在用的文章最优参数，建议别改了
- **`backtest.py`**: 回测框架，包括SKA和AKA
- **`visualizations.py`**: 画图，目前把前四章图出了

---

**建议：** 改代码在上面五个文件里改，model.py和model.ipynb都是不完善的版本，而且过分冗长。
