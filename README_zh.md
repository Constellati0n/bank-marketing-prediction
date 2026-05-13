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
│   ├── main.py           # LightGBM 主模型流程
│   └── baseline.py       # 线性回归基线 + 可视化
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

- 5 折分层交叉验证 + 早停（patience=50）
- 通过 `scale_pos_weight` 处理类别不平衡
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
```

## 模型评估

数据集严重不平衡（86.9% "no" vs 13.1% "yes"），因此准确率具有误导性——简单的"全部预测 no"就能得到 86.9% 的准确率。真正的挑战在于识别少数正类样本。

### 模型对比

| 指标 | 线性回归（基线） | LightGBM（5 折 CV） |
|------|------------------|---------------------|
| ROC-AUC | — | 0.797 |
| 准确率 (Accuracy) | 86.44% | — |
| 精确率 (Precision) | 45.98% | 41.79% |
| 召回率 (Recall) | 13.42% | 58.74% |
| F1 分数 | 20.78% | 48.84% |

### 关键解读

**ROC-AUC = 0.80** 证明模型具备真实的区分能力——它能比随机猜测好得多地识别潜在订阅客户。

**默认 0.5 阈值在不平衡数据上失效**。由于 `scale_pos_weight=6.62` 压低了原始输出概率，没有样本超过 0.5 → LightGBM 默认全判 "no"。

**阈值调优后**（网格搜索最优阈值 = 0.250，以 F1 最大化为目标），模型成功捕获 **58.7% 的真实订阅客户**——是基线模型 13.4% 的 **4.4 倍**。F1 分数翻倍以上（0.21 → 0.49）。

这对应真实营销场景：~42% 的精确率意味着每 10 个被标记的客户中有 4 个会转化，~59% 的召回率意味着覆盖了大多数潜在客户。营销团队会更看重这种折衷而非基线模型。

### 输出文件

| 文件 | 模型 | 说明 |
|------|------|------|
| `output/submission_result.csv` | LightGBM | 5 折交叉验证 + 阈值调优预测 |
| `output/submission_baseline.csv` | 线性回归 | 单次划分预测 |

可视化图表保存至 `figures/`：

## 技术栈

- **Python 3.8+**
- **LightGBM** — 梯度提升模型
- **scikit-learn** — 预处理、评估指标、管道
- **Pandas / NumPy** — 数据处理
- **Matplotlib / Seaborn** — 可视化
