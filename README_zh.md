# 银行营销订阅预测系统

[English](README.md) | 中文

基于 LightGBM 与 5 折分层交叉验证，预测银行客户是否会订阅定期存款。

## 数据集

基于 [UCI Bank Marketing 数据集](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)，包含葡萄牙某银行电话营销活动的客户数据。

| 文件 | 说明 |
|------|------|
| `data/train.csv` | 训练集，含目标变量 `subscribe` |
| `data/test.csv` | 测试集，不含目标变量 |
| `data/submission.csv` | 提交模板 |

### 特征概览

**客户信息**：age, job, marital, education, default, housing, loan

**营销联系**：contact, month, day_of_week, duration, campaign, pdays, previous, poutcome

**经济指标**：emp_var_rate, cons_price_index, cons_conf_index, lending_rate3m, nr_employed

## 项目结构

```
bank-marketing-prediction/
├── src/
│   ├── main.py           # LightGBM 主模型流程（已优化）
│   ├── baseline.py       # 线性回归基线 + 可视化
│   └── optimize.py       # 超参数网格搜索（11 组配置）
├── notebooks/
│   └── eda.ipynb         # 探索性数据分析
├── data/                 # 数据集
├── figures/              # 可视化图表
├── output/               # 预测结果
├── requirements.txt
├── .gitignore
└── README.md
```

## 特征工程

| 特征 | 构造方式 | 业务含义 |
|------|----------|----------|
| `contacted_before` | `pdays == 999 → 0, 否则 1` | 客户是否曾被联系过 |
| `month_sin/cos` | 周期性编码 | 捕捉月份季节性，避免虚假序数关系 |
| `day_sin/cos` | 周期性编码 | 捕捉星期几的周期性 |
| `total_contacts` | `campaign + previous` | 跨活动总联系次数 |
| `emp_cons_ratio` | `emp_var_rate / cons_conf_index` | 经济指标交互特征 |
| `age_group` | 分箱：young/mid/senior/elder | 捕捉年龄的非线性效应 |
| `high_contact_freq` | `campaign > 5 → 1` | 标记过度联系的客户 |
| `previous_success` | `poutcome == success → 1` | 历史营销成功指标 |

## 模型说明

### 主模型：LightGBM

- **参数已优化**（通过 `src/optimize.py` 对 11 组配置进行网格搜索）
- 5 折分层交叉验证 + AUC 早停（patience=30）
- `scale_pos_weight=1.0` — 保留自然概率分布，不平衡通过阈值调优处理
- 深层树：`num_leaves=63`、`learning_rate=0.03`、`min_child_samples=10`
- 预处理管道：`SimpleImputer` + `StandardScaler`（数值特征）、`SimpleImputer` + `OneHotEncoder`（分类特征）

### 基线模型：线性回归

- 80/20 训练验证划分
- 生成诊断图表：特征重要性、残差图、PR 曲线、混淆矩阵

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 LightGBM 主模型
python src/main.py

# 运行基线模型 + 可视化
python src/baseline.py

# 运行超参数搜索（11 种配置）
python src/optimize.py
```

## 模型评估

数据集严重不平衡（86.9% "no" vs 13.1% "yes"），因此准确率具有误导性——简单的"全部预测 no"就能得到 86.9% 的准确率。真正的挑战在于识别少数正类样本。

### 模型对比

| 指标 | 线性回归（基线） | LightGBM v1 | LightGBM v2（优化后） |
|------|------------------|:-----------:|:---------------------:|
| ROC-AUC | — | 0.797 | **0.791** |
| 精确率 (Precision) | 45.98% | 41.79% | **44.24%** |
| 召回率 (Recall) | 13.42% | 58.74% | **56.10%** |
| F1 分数 | 20.78% | 48.84% | **49.47%** |

### 优化历程

**v1 — 初始方案**：`scale_pos_weight=6.62` + `binary_logloss` 评估指标 + 浅层树（`num_leaves=31`）。模型在第 5 轮即早停——因为 log-loss 在不平衡数据上几乎瞬时收敛，模型默认判全 "no"，全靠阈值调优才挽救回来。

**v2 — 优化后**：通过 `src/optimize.py` 对 11 组参数进行网格搜索。关键改动：

| 参数 | v1 | v2 | 优化理由 |
|------|:--:|:--:|----------|
| `metric` | `binary_logloss` | `auc` | AUC 对不平衡数据更有效；log-loss 倾向全判 no |
| `scale_pos_weight` | 6.62 | 1.0 | 降低权重使模型输出更宽的概率分布 |
| `learning_rate` | 0.05 | 0.03 | 深层树需要更小的步长 |
| `num_leaves` | 31 | 63 | 更大容量捕捉非线性模式 |
| `min_child_samples` | 20 | 10 | 允许从少数类小样本组中学习 |
| 早停轮数 | ~5 | ~30-70 | AUC 不会在不平衡数据上假收敛 |

**优化效果**：精确率提升 +2.5 个百分点（41.8% → 44.2%），召回率仅损失 -2.6 个百分点，F1 略有提升。真正的价值在于：系统化的实验比一次性的参数猜测更有效——而且正确的评估指标比调参本身更重要。

### 输出文件

| 文件 | 模型 | 说明 |
|------|------|------|
| `output/submission_result.csv` | LightGBM（优化后）| 5 折交叉验证 + 阈值调优 |
| `output/submission_baseline.csv` | 线性回归 | 单次划分预测 |
| `output/submission_optimized.csv` | LightGBM（最优组合）| `src/optimize.py` 网格搜索结果 |

可视化图表保存至 `figures/`：

## 技术栈

- **Python 3.8+**
- **LightGBM** — 梯度提升模型
- **scikit-learn** — 预处理、评估指标、管道
- **Pandas / NumPy** — 数据处理
- **Matplotlib / Seaborn** — 可视化
