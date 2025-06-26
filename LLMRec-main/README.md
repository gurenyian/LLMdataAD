# LLM推荐技术创新实验框架

本项目实现了基于大语言模型(LLM)的推荐系统数据增强创新方案，包含分布自适应调控和对抗性质量保证两大核心模块。

## 🎯 核心创新点

1. **分布自适应调控模块**
   - 数据分布分析器：多维度分析真实数据特征
   - 动态Prompt调优器：智能调整生成策略

2. **对抗性质量保证模块**
   - 生成器-判别器博弈机制
   - 自我反思与质量筛选

3. **创新评估指标**
   - 对抗鉴别成功率
   - 长尾激活度
   - 泛化鲁棒性

## 🏗️ 项目结构

```
LLMRec-Innovation/
├── data/                           # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   └── results/                   # 实验结果
├── core/                          # 核心模块
│   ├── data_distribution_analyzer.py
│   ├── dynamic_prompt_tuner.py
│   ├── adversarial_quality_module.py
│   └── evaluation_metrics.py
├── experiments/                   # 实验脚本
│   ├── main_experiment.py
│   ├── run_complete_experiment.py
│   └── config.py
├── utils/                         # 工具模块
│   ├── logger.py
│   └── data_utils.py
└── requirements.txt
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd LLMRec-Innovation

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 方式1: 直接运行完整实验
cd experiments
python run_complete_experiment.py

# 方式2: 交互式运行
python run_complete_experiment.py
# 然后根据提示选择实验模式
```

### 3. 实验模式选择

- **完整创新实验 (推荐)**: 5轮迭代，100样本/轮
- **快速验证实验**: 3轮迭代，50样本/轮
- **自定义配置实验**: 用户自定义参数

## 📊 实验流程

1. **Phase 1: 数据分析**
   - 分析真实数据的分布特征
   - 生成量化的特征向量

2. **Phase 2: 迭代优化**
   - 生成合成数据
   - 对抗性质量保证
   - 分布偏差分析
   - 动态Prompt调优

3. **Phase 3: 综合评估**
   - 对抗鉴别成功率评估
   - 长尾激活度分析
   - 泛化鲁棒性测试

4. **Phase 4: 结果整合**
   - 生成详细实验报告
   - 保存结果和可视化

## 📈 评估指标

### 传统指标
- NDCG@k
- Recall@k
- 精确率和召回率

### 创新指标
- **对抗鉴别成功率**: 生成器欺骗判别器的能力
- **长尾激活度**: 长尾物品的推荐激活程度
- **泛化鲁棒性**: 跨域和时间的稳定性

## 🛠️ 核心模块说明

### DataDistributionAnalyzer
- 用户活跃度分布分析
- 物品流行度基尼系数计算
- 会话长度模式识别
- 用户类型聚类分析

### DynamicPromptTuner
- 分布偏差精准计算
- 结构化Prompt库管理
- 智能策略选择与组合
- 生成质量持续优化

### AdversarialQualityModule
- 判别器训练与评估
- 生成器自我反思机制
- 质量评估与样本筛选
- 对抗训练历史追踪

### InnovativeEvaluationMetrics
- 多维度创新指标计算
- 鲁棒性测试套件
- 长尾效应量化分析
- 综合评估报告生成

## 📁 结果输出

实验完成后，结果保存在 `data/results/` 目录下：

- `innovation_experiment_[timestamp].json`: 完整实验数据
- `logs/`: 运行日志文件

## 🔧 配置说明

在 `experiments/config.py` 中可以调整：

- 迭代次数和样本数量
- 收敛阈值
- 质量改进阈值
- 各模块详细参数



## 🙏 致谢

感谢所有为推荐系统和大语言模型研究做出贡献的研究者们。

## 📞 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。
