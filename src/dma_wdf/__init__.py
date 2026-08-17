"""DMA-WDF：供水 DMA 时空预测工具包。

基于以下论文的数据预处理管道：
*"Water demand forecasting in multiple district metered areas based on a
multi-scale correction module neural network architecture"*
— Water Research X, 2024, Article 100269.

模块结构
--------
- ``data`` : 数据处理全流程（加载 → 清洗 → 特征 → 张量）
- ``models`` : 模型实现（预留）
- ``quality`` : 数据质量检查与论文协议验证
- ``utils`` : 通用工具（配置加载等）
"""
