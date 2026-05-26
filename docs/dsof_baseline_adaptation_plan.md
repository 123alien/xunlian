# 2026-05-19 状态说明：本文件为 DSOF 路线归档记录，不是当前论文主线

当前论文主线已经从 DSOF/PA-DSOF 和早期 PARBoost 路线收敛到 **PA-MSR / PA-MSR+**：

- **PA-MSR** = persistence anchor + residual RevIN + multi-scale residual encoding，对应 `M9_REVIN_MSR`。
- **PA-MSR+** = PA-MSR + daily-profile source selection + validation calibration，对应 `M11_REVIN_MSR_DPS_CAL`。
- DSOF/PA-DSOF 只保留为“曾经考虑过的顶会 baseline 复现与适配路线”，不作为本文最终方法或主 claim。
- PARBoost 只保留为“早期 persistence-anchored residual boosting baseline”和补充/历史证据，不作为最终提出方法。
- 当前主文写作应以 `docs/manuscript_draft_parboost.md` 中的 PA-MSR 证据链为准，尤其是 `results/final_claim_evidence/` 下的最终表格。

因此，下面旧计划中的 DSOF、PA-DSOF、模块缝合、以及“基于 DSOF 提出方法”的表述，均不应直接进入论文主文。若引用，只能作为研究路线探索或失败/未采用方案记录。

# DSOF 顶会 baseline 复现与建筑冷启动适配路线

## 目标

用 ICLR 2025 的开源方法 **DSOF** 作为一个新的顶会 baseline，先把官方代码跑通并复现出最小结果，再把它适配到我们的冷启动建筑电力负荷预测任务里。

这条路线的论文主张应该是：

> 通用 online time-series forecasting 方法能缓解流式预测评估中的信息泄漏问题，但冷启动建筑负荷预测还有更特殊的问题：强 persistence、建筑间异质性、source-target mismatch、few-shot 不稳定。本文在 DSOF 的 dual-stream online forecasting 框架基础上，提出面向建筑负荷的 persistence-anchored residual adaptation 和 similarity-aware source replay，以降低负迁移和 worst-case error。

注意：这个不能写成“我们偷偷换模块”。更稳、更诚实的写法是：

> 我们复现并系统评估了一个最新顶会 online forecasting baseline，然后针对建筑冷启动场景进行领域适配。

这样审稿人问来源、问创新、问为什么要改，都能回答。

## 选定的 baseline

- 论文：*Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage*
- 会议：ICLR 2025
- 作者：Ying-yee Ava Lau, Zhiwen Shao, Dit-Yan Yeung
- 官方代码：https://github.com/yyalau/iclr2025_dsof
- 本地代码：`external/iclr2025_dsof`
- 实验机代码：`~/桌面/xunlian/external/iclr2025_dsof`
- 实验机环境：conda env `dsof`，Python 3.8，PyTorch 2.1.0 + cu118

## 当前复现状态

官方 ETTh2 小复现已经在实验机跑通。

运行命令：

```bash
cd ~/桌面/xunlian/external/iclr2025_dsof
source /home/lrh/miniconda3/etc/profile.d/conda.sh
conda activate dsof
CUDA_VISIBLE_DEVICES=0 python -u src/main.py \
  --itr 1 \
  --pred_len 1 \
  --data ETTh2 \
  --num_workers 0 \
  --y_model_main DLinear \
  --y_model_student MLP \
  --y_opt w_student/DLinear_MLP/ETTh2 \
  --y_trainer w_student/residual/dsof \
  --comments full_dsof_etth2_pl1
```

已经得到的结果：

| 设置 | MAE | MSE |
|---|---:|---:|
| DSOF, ETTh2, pred_len=1 | 0.3479 | 0.3667 |
| DSOF, ETTh2, pred_len=24 | 0.5895 | 1.7045 |
| DSOF, ETTh2, pred_len=48 | 0.7139 | 3.1177 |
| ER only, ETTh2, pred_len=1 | 0.3505 | 0.3800 |
| TD only, ETTh2, pred_len=1 | 0.3576 | 0.3979 |
| residual baseline, ETTh2, pred_len=1 | 0.3917 | 0.5124 |

关键结论：

- 官方代码训练、测试、checkpoint、结果保存链路已经跑通。
- DSOF 在 ETTh2 `pred_len=1` 上优于 ER only、TD only 和 residual baseline。
- 这只是最小复现，不是完整复现论文全部结果。

输出目录示例：

```text
~/桌面/xunlian/external/iclr2025_dsof/exps/
```

## DSOF 模块审计

### 1. Teacher model：基础预测器

相关文件：

```text
src/models/DLinear.py
```

作用：

- DSOF 里用 teacher model 产生基础预测。
- 当前跑通的是 `DLinear`。
- DLinear 会把序列分成 seasonal 和 trend 两部分，再用线性层预测未来。

对我们有什么用：

- 它是一个轻量、可解释的神经预测器。
- 我们之前已经用过 DLinear，所以它和现有论文线索能接上。
- 但在建筑负荷里，它不能替代 Persistence，因为 Persistence 本身太强，必须显式作为基准或 anchor。

### 2. Student model：残差修正器

相关文件：

```text
src/models/MLP.py
```

作用：

- Student model 接收历史序列和 teacher 的未来预测。
- 它学习一个 correction / residual。
- 在 residual DSOF 里，最终预测是 teacher prediction 加 student correction。

对我们有什么用：

- 这是 DSOF 和 PARBoost 最能连接的地方。
- 建筑任务里，student 不应该只是修正 DLinear，而应该修正 persistence-anchored forecast。

### 3. Residual combination：teacher + student 的残差组合

相关文件：

```text
src/trainers/w_student/trainerFramework_ERTDRes.py
```

核心逻辑：

```python
final_prediction = teacher_prediction + student_prediction
```

如果维度不同，则使用：

```python
final_prediction = teacher_future + student_prediction
```

作用：

- 把基础预测和残差修正分开。
- teacher 学一个较稳定的全局预测。
- student 学在线更新后的局部修正。

我们的建筑适配方式：

```text
y_hat[t+h] = y[t] + r_teacher[t+h] + r_student[t+h]
```

其中：

```text
r[t+h] = y[t+h] - y[t]
```

也就是说，不直接预测未来负荷绝对值，而是预测相对 persistence 的 residual。

这是最关键的改造点。

### 4. Slow stream：experience replay

相关文件：

```text
src/trainers/trainerBaseERTD.py
```

对应方法：

```python
er(...)
```

作用：

- 维护一个 replay buffer。
- 在测试/在线阶段，从过去完整窗口中采样 replay 数据。
- 通过 replay 防止模型只记住最新的噪声。

对建筑任务的问题：

- 原始 DSOF 的 replay 是时间上的 replay。
- 冷启动建筑迁移里，我们更关心 source building replay。
- 如果 source building 和 target building 很不相似，replay 反而可能造成负迁移。

我们的适配方式：

- 用目标建筑 few-shot adaptation window 计算 similarity。
- 只选相似 source building 进入 replay。
- similarity 不能用 test set，否则就是数据泄漏。

候选版本：

| 方法名 | 含义 |
|---|---|
| DSOF | 原始 DSOF |
| Building-DSOF | 直接把 DSOF 跑在建筑数据上 |
| Sim-DSOF | 使用 similarity-aware source replay |
| PA-DSOF | persistence anchoring + similarity-aware replay |

### 5. Fast stream：temporal difference online update

相关文件：

```text
src/trainers/trainerBaseERTD.py
```

对应方法：

```python
td(...)
```

作用：

- 在真实在线预测中，有些未来点会逐步变成已观测值。
- TD 用已经观测到的一部分未来值和当前模型伪预测构造更新目标。
- 目的是快速适应最近分布变化，同时避免直接用未发生的未来标签。

对建筑任务的意义：

- 适合写成“冷启动之后的在线校准”。
- 但是必须非常小心信息泄漏。

我们需要分清两个评估设置：

| 设置 | 是否允许 test 期间在线更新 |
|---|---|
| strict cold-start | 不允许用 test 标签更新 |
| online adaptation | 可以在某个时刻标签被评分之后，用它更新后续预测 |

论文里不能混着写，否则审稿人会说 online update 偷看了 test label。

### 6. No-leakage protocol：无泄漏评估协议

DSOF 原论文的一个核心动机就是指出旧 online forecasting protocol 有信息泄漏风险。

我们必须把这一点继承到建筑任务里：

- 目标建筑 few-shot days 可以用于 adaptation。
- test days 必须按时间顺序完全留出。
- similarity 只能用 source training data 和 target few-shot data。
- scaler、normalizer、calibration 不能 fit 在 test data 上。
- 如果 online adaptation 使用 test 期间逐步到来的标签，必须先评分，再用于未来更新。

这一点会成为我们论文实验可信度的关键。

## 我们的方法方向

暂定方法名：

```text
PA-DSOF
```

全称：

```text
Persistence-Anchored Dual-Stream Online Forecasting
```

核心预测形式：

```text
y_hat[t+h] = y[t] + r_teacher[t+h] + r_student[t+h]
```

其中：

```text
r[t+h] = y[t+h] - y[t]
```

也可以先做一个更简单的版本：

```text
y_hat[t+h] = y[t] + r_student[t+h]
```

这个版本更接近 PARBoost，也更容易先验证 residual anchoring 是否有效。

## Similarity-aware source replay

相似性只能从目标建筑 few-shot 数据计算。

可以使用的特征：

- few-shot mean load
- few-shot load std
- zero ratio
- daily profile shape
- lag-1 persistence MAE
- residual variance
- hour-of-day profile

不能使用：

- target test mean
- target test zero ratio
- target test daily profile
- 任何 test period 的统计量

推荐 replay 权重：

```text
w(source, target) = similarity(source_features, target_fewshot_features)
```

或者更保守：

```text
只选择 top-s most similar source buildings
```

## 实验矩阵

### Stage 1：官方 baseline 复现

目的：

- 证明 DSOF 官方代码能跑。
- 拿到一个真实顶会 baseline 的复现入口。

最低要求：

- ETTh2, pred_len = 1, 24, 48
- DSOF
- ER only
- TD only
- residual baseline

目前状态：

- ETTh2 已经跑通。
- pred_len=1/24/48 的 DSOF 已完成。
- ER/TD/baseline 的 pred_len=1 已完成。

### Stage 2：原始 DSOF 跑建筑数据

目的：

- 看原始 DSOF 在建筑冷启动上到底行不行。
- 如果它不稳，我们的方法动机就成立。

数据集：

- BDG2-24
- BDG2-120 active robustness
- COFACTOR-44

对照方法：

- Persistence
- RF-ST
- ExtraTrees-ST
- DLinear-M3
- PARBoost
- original DSOF

指标：

- mean MAE
- median MAE
- worst-case MAE
- mean rank
- top-3 rate
- negative transfer rate
- win rate vs Persistence
- win rate vs RF-ST
- win rate vs ExtraTrees-ST

### Stage 3：建筑适配版 PA-DSOF

目的：

- 证明不是简单换模型，而是建筑领域问题驱动的适配。

方法：

- original DSOF
- DSOF + persistence residual target
- DSOF + similarity-aware replay
- PA-DSOF full
- PARBoost

必须有的消融：

- 去掉 persistence anchor
- 去掉 similarity replay
- random source replay
- all-source replay
- target-only online adaptation

## 论文 claim 和证据对应

### Claim 1：DSOF 是合理的现代顶会 baseline

需要证据：

- 官方代码能跑。
- 官方数据 ETTh2 有复现结果。
- 保存了 predictions、metrics、环境和命令。

当前状态：

- 已基本完成。

### Claim 2：原始 DSOF 不能直接解决建筑冷启动问题

需要证据：

- original DSOF 在 BDG2 / COFACTOR 上的结果。
- 必须和 Persistence、RF-ST、ExtraTrees-ST 比。
- 不能只和神经网络比。

当前状态：

- 还没跑，这是下一步。

### Claim 3：persistence anchoring 是建筑任务里的关键机制

需要证据：

- absolute target vs residual target 消融。
- 看 mean MAE，更要看 worst-case MAE 和 negative transfer。

当前状态：

- PARBoost 已经有强证据。
- 还需要在 DSOF 框架里验证。

### Claim 4：similarity-aware replay 能降低 source-target mismatch

需要证据：

- similarity replay vs all-source replay vs random replay。
- mismatch 高的建筑上应该改善更明显。

当前状态：

- 还没做。

## 审稿人风险

### 风险 1：这是不是 DSOF 小改？

应对：

- 不把贡献写成“全新通用时间序列模型”。
- 写成“面向建筑冷启动负荷预测的 DSOF 领域适配与鲁棒性增强”。
- 重点强调建筑任务中特有的 persistence dominance 和 source-target mismatch。

### 风险 2：为什么不和强树模型比？

应对：

- 所有主表必须保留 Persistence、RF-ST、ExtraTrees-ST。
- 不要只挑神经网络做对比。

### 风险 3：online update 是否使用了 test label？

应对：

- strict cold-start 和 online adaptation 分开报告。
- online setting 必须遵守“先评分，后更新未来预测”。

### 风险 4：similarity 是否用了 test set？

应对：

- 明确 similarity 只由 target few-shot window 计算。
- 保存 similarity feature table，方便审计。

### 风险 5：BDG2 subset 是否 cherry-pick？

应对：

- BDG2-24 用作方法开发。
- BDG2-120 active robustness 用作规模验证。
- COFACTOR-44 用作外部验证。

## Go / No-Go 标准

可以继续写方法论文，如果：

- PA-DSOF 明显优于 original DSOF。
- PA-DSOF 在 BDG2-24 和 BDG2-120 上降低 worst-case MAE 或 negative transfer。
- PA-DSOF 大多数 active buildings 上赢 Persistence。
- COFACTOR 上即使不赢 ExtraTrees-ST，也至少稳定优于 original DSOF 和 Persistence。

需要转成分析型论文，如果：

- PA-DSOF 不如 original DSOF。
- similarity replay 不如 random/all-source replay。
- 只在 BDG2-24 有效，在 BDG2-120 上消失。
- COFACTOR 上没有任何稳定收益。

如果 no-go，论文可以转成：

```text
When Do Online Transfer Forecasting Methods Fail in Cold-Start Building Load Prediction?
```

## 立刻要做的下一步

1. 把 BDG2 / COFACTOR 转成 DSOF 能读取的 CSV 格式。
2. 先跑 original DSOF on BDG2-24。
3. 汇总 original DSOF 与 Persistence、RF-ST、ExtraTrees-ST、PARBoost 的对比。
4. 如果 original DSOF 明显不稳，开始实现 persistence residual target。
5. residual DSOF 有收益后，再做 similarity-aware replay。

## 模块筛选与顶会模块适配计划

### 已经验证有用的内部模块

根据 `results/pa_dsof_module_pilot_bdg2/aggregate/table_module_ablation_summary.csv`，当前 BDG2-24 pilot 的结论是：

| 模块 | 证据 | 判断 |
|---|---|---|
| Persistence anchoring | direct source-target MAE 约 17-19，加入 residual anchor 后降到 2-3 | 必须保留，主贡献模块 |
| Similarity-aware replay | k=7/14/30 有小幅收益，k=3 不稳定 | 可保留，但不能夸大 |
| Calibration | k=30 最明显，k=3/7/14 作用较小 | 作为稳定化模块，不作为主创新 |
| Full combination | 总体稳，但不是所有 k 第一 | 需要进一步调权或重构 |

最重要的判断：

```text
真正有效的是 persistence anchoring。
similarity replay 和 calibration 是辅助模块。
```

因此后续“缝合”不能乱加模块，而应该围绕三个真实问题：

1. 建筑负荷分布漂移：不同建筑、不同时间段均值/尺度差异大。
2. source-target mismatch：不相似源建筑会造成负迁移。
3. few-shot 不稳定：k=3 时目标建筑信息太少，相似性估计容易失真。

### 顶会候选模块 1：RevIN

来源：

- RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift
- ICLR 2022
- 官方代码：`https://github.com/ts-kim/RevIN`

模块作用：

- 对每个样本/实例做可逆归一化。
- 先移除实例级均值和尺度，再预测，最后把统计量还原。
- 主要针对 distribution shift。

为什么适合我们：

- 建筑之间的负荷尺度差异非常大。
- BDG2 有 near-zero/inactive building，也有高负荷建筑。
- source-target mismatch 的一个主要来源就是尺度和均值不一致。

建议适配方式：

```text
PA-DSOF + Residual-RevIN
```

不是对原始负荷直接 RevIN，而是对 residual target 做 RevIN：

```text
r[t+h] = y[t+h] - y[t]
```

然后：

```text
normalize residual sequence -> model -> denormalize residual -> y[t] + residual
```

要做的消融：

| 方法 | 含义 |
|---|---|
| PA-DSOF | 原方法 |
| PA-DSOF + input RevIN | 对输入负荷做 RevIN |
| PA-DSOF + residual RevIN | 对 residual 做 RevIN |
| PA-DSOF + source-fitted scaler | 传统 source scaler 对照 |

优先级：

```text
最高。
```

因为它和我们已有的 persistence anchoring 不冲突，而且非常容易解释。

### 顶会候选模块 2：Non-stationary Transformer 的 stationarization / de-stationary 思路

来源：

- Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting
- NeurIPS 2022
- 官方代码：`https://github.com/thuml/Nonstationary_Transformers`

模块作用：

- Series Stationarization：让输入序列更平稳，降低预测难度。
- De-stationary Attention：避免过度平稳化，把原始非平稳信息补回注意力。

为什么适合我们：

- 建筑负荷有明显的工作日/周末、运行状态、设备启停变化。
- 直接神经迁移容易因为非平稳性崩掉。
- 我们的 residual anchoring 本质上已经是一种弱 stationarization。

建议适配方式：

```text
Persistence anchoring = first-order stationarization
Residual gate = de-stationary correction
```

不要直接搬完整 Non-stationary Transformer，太重。我们可以做一个轻量模块：

```text
residual_hat = gate(context) * model_residual
```

其中 gate 使用 few-shot 统计量：

- residual variance
- zero ratio
- hour-of-day profile strength
- persistence MAE

直觉：

- 如果 persistence 已经非常强，gate 自动减小 correction。
- 如果负荷变化明显，gate 放大 residual correction。

方法名可以写成：

```text
Persistence-aware de-stationary residual gate
```

要做的消融：

| 方法 | 含义 |
|---|---|
| PA-DSOF | 无 gate |
| PA-DSOF + scalar gate | 每栋建筑一个 correction 强度 |
| PA-DSOF + time-varying gate | 每个 hour/context 一个 correction 强度 |
| PA-DSOF + oracle gate | 只作为上界，不进主表 |

优先级：

```text
高。
```

因为它能解释为什么 k=3 时 similarity replay 不稳：few-shot 信息少时 correction 应该被压小。

### 顶会候选模块 3：PatchTST 的 patching + channel independence

来源：

- A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- ICLR 2023
- 官方代码：`https://github.com/yuqinie98/PatchTST`

模块作用：

- 把时间序列切成 patch，而不是逐点建模。
- channel-independent 设计让每个变量共享模型结构但独立建模。
- 减少注意力计算量，也让长历史模式更容易学习。

为什么适合我们：

- 建筑负荷有小时级噪声，但日周期/运行周期更重要。
- few-shot 下逐点学习容易过拟合。
- patch 可以把 24 小时或 12 小时作为更稳定的局部模式。

建议适配方式：

```text
Residual Patch Encoder
```

不是直接上完整 PatchTST，而是在 residual 序列上提取 patch features：

```text
past residual sequence -> 24h patches -> patch embedding -> residual predictor
```

可以先做很轻的版本：

- 24h patch mean
- 24h patch std
- 24h patch slope
- last patch residual profile

再做神经版本：

- PatchTST encoder 替换 MLP student

要做的消融：

| 方法 | 含义 |
|---|---|
| PA-DSOF + tabular residual features | 当前版本 |
| PA-DSOF + handcrafted patch features | 轻量 patch |
| PA-DSOF + PatchTST student | 神经 patch |

优先级：

```text
中高。
```

它对提升神经版本故事性有用，但不一定比树模型更强。

### 顶会候选模块 4：iTransformer 的 variate-token 表示

来源：

- iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- ICLR 2024 Spotlight
- 官方代码：`https://github.com/thuml/iTransformer`

模块作用：

- 把变量维度当作 token，让 attention 在变量之间建模。
- 更适合高维多变量时间序列。

为什么可能适合我们：

- 如果未来加入天气、日历、建筑元数据、多源建筑特征，变量间关系会更重要。

为什么不建议第一批做：

- 当前我们的单建筑负荷预测变量并不多。
- iTransformer 更像 backbone 替换，故事容易变成“换了个网络”。
- 它不能直接解决 persistence dominance。

优先级：

```text
低到中。
```

可以作为后续 neural backbone，不作为第一批核心模块。

## 推荐的“缝合”路线

不要一次塞 4 个模块。建议只做两条主线：

### 路线 A：最稳论文路线

```text
Persistence anchoring
+ Residual-RevIN
+ Similarity-aware replay
+ Residual gate
```

对应论文故事：

> 建筑冷启动预测的核心困难是 persistence dominance、跨建筑分布漂移和 source-target mismatch。我们用 persistence anchoring 降低预测难度，用 residual RevIN 处理分布漂移，用 similarity-aware replay 控制迁移来源，并用 residual gate 防止 few-shot 下过度修正。

这条路线最像 Energy and Buildings / Journal of Building Engineering 能接受的工程型方法论文。

### 路线 B：神经模型更强故事路线

```text
Persistence anchoring
+ Residual Patch Encoder
+ DSOF fast-slow online update
+ Similarity-aware replay
```

对应论文故事：

> 通用 online forecasting 方法没有针对建筑负荷的强周期 residual 结构。我们把 residual 序列 patch 化，并在 dual-stream online adaptation 中进行相似源 replay。

这条更像 ML 方法，但风险更大，因为神经模型未必赢树模型。

## 当前推荐

优先做路线 A。

第一批只跑三个新增模块：

1. `PA + Residual-RevIN`
2. `PA + Residual Gate`
3. `PA + Residual-RevIN + Residual Gate + Similarity Replay`

最小 pilot：

- BDG2-24
- k = 3, 7, 14, 30
- seed = 42
- 对照：
  - Direct source-target
  - Persistence anchor
  - Persistence anchor + similarity replay
  - Persistence anchor + residual RevIN
  - Persistence anchor + residual gate
  - Full

Go / No-Go：

- 如果 `Residual-RevIN` 在 k=3 改善明显，说明它解决 few-shot 分布漂移。
- 如果 `Residual Gate` 降低 worst-case MAE，说明它解决过度修正。
- 如果 Full 比单独 persistence anchor 只提升很小，就不要把模块堆成主贡献，改写成“persistence anchoring 为主，其他模块为稳定化辅助”。

## 2026-05-19：证据链收敛版

本节覆盖上面的早期设想。当前实验已经说明：`Residual Gate` 不应继续作为主线；普通统计 similarity 也不应作为核心贡献。论文主线应收窄为一个面向建筑冷启动负荷预测的 **persistence-anchored residual transfer framework**，核心有效模块是 `RevIN` 和 `MSR`，`DPS + Calibration` 只作为中等冷启动条件下的可选增强。

### 收窄后的中心 claim

推荐论文中心主张：

```text
Cold-start building load forecasting is strongly affected by persistence dominance,
cross-building distribution shift, and source-target mismatch. We propose a
persistence-anchored residual transfer framework that combines reversible residual
normalization and multi-scale residual encoding. Across BDG2 and COFACTOR, RevIN
and MSR provide consistent paired improvements over the persistence-anchored
baseline, while daily-profile source selection and calibration are beneficial
mainly under moderate cold-start settings.
```

中文表述：

```text
冷启动建筑负荷预测受到强 persistence 基线、跨建筑分布漂移和源-目标建筑不匹配的共同影响。
本文提出一个 persistence-anchored residual transfer 框架，并通过 residual RevIN 和
multi-scale residual encoding 提高冷启动预测稳定性。BDG2 和 COFACTOR 上的配对实验表明，
RevIN 与 MSR 能带来稳定改善；DPS 和 calibration 在 k=14/30 等目标数据稍多的场景下进一步有效，
但在 k=3 极端冷启动下不应过度使用。
```

这比“提出一个全面最优的新模型”更窄，但更能被当前结果支撑。

### 当前证据库存

| 证据 | 结果文件 | 结论 |
|---|---|---|
| COFACTOR-44 候选验证 | `results/pa_module_extensions_cofactor_44/aggregate/table_pa_module_extensions_summary.csv` | `M3_ANCHOR_REVIN` 在 k=3/7 最好，`M7_FULL_REVIN_SIM_CAL` 在 k=14/30 最好 |
| COFACTOR-20 无 Gate 筛选 | `results/pa_module_screen_cofactor_20_nogate/aggregate/table_pa_module_extensions_summary.csv` | 复现 COFACTOR-44 趋势；普通 similarity 单独不可靠 |
| COFACTOR-12 MSR/DPS | `results/pa_module_screen_cofactor_12_msr_dps/aggregate/table_pa_module_extensions_summary.csv` | `M9_REVIN_MSR` 在 k=3/7 最好，`M11_REVIN_MSR_DPS_CAL` 在 k=14/30 最好 |
| BDG2-24 MSR/DPS | `results/pa_module_screen_bdg2_24_msr_dps/aggregate/table_pa_module_extensions_summary.csv` | MSR/DPS 信号能跨回 BDG2；`M9` 和 `M11` 均超过 `M3` |

### 模块级判断

| 模块 | 对应方法 | 当前判断 | 可写 claim |
|---|---|---|---|
| Persistence anchor | `M1_ANCHOR` | 必须保留，是任务重定义，不是最终充分方法 | 将强 persistence baseline 显式纳入预测公式，避免模型与 persistence 正面对抗 |
| RevIN | `M3_ANCHOR_REVIN` | 显著有效，最稳模块 | 缓解跨建筑尺度和分布漂移 |
| MSR | `M9_REVIN_MSR` | 显著有效，最值得作为新增模块 | 捕捉小时级、半日、日周期等多尺度 residual structure |
| DPS | `M10_REVIN_MSR_DPS` | 单独效果不稳定 | 不能单独夸大，只能作为 source-target mismatch 的领域化相似度设计 |
| DPS + Calibration | `M11_REVIN_MSR_DPS_CAL` | k=14/30 有价值，k=3/7 不稳 | 在目标 few-shot 较多时提升 source selection 与 residual correction 的可靠性 |
| 普通 similarity | `M2_ANCHOR_SIM`, `M5_ANCHOR_REVIN_SIM` | 不显著甚至有害 | 不作为主贡献，可作为被替代的弱 baseline |
| Gate | `M4`, `M6` | 删除 | 不进入主文方法，不作为正向模块 |

### 配对显著性证据

当前最能写进论文的统计证据：

1. `M3_ANCHOR_REVIN` vs `M1_ANCHOR`：
   - COFACTOR-44 上 k=3/7/14/30 均显著改善；
   - 胜率约 95%-100%；
   - Wilcoxon p < 0.0001。

2. `M9_REVIN_MSR` vs `M3_ANCHOR_REVIN`：
   - COFACTOR-12 上 k=3/7/14/30 均显著改善；
   - BDG2-24 上 k=3/7/14/30 均显著改善；
   - 说明 MSR 的收益不是单数据集偶然现象。

3. `M11_REVIN_MSR_DPS_CAL` vs `M3_ANCHOR_REVIN`：
   - COFACTOR-12 上 k=14/30 显著；
   - k=3/7 不显著；
   - 因此只能写成 “moderate cold-start settings 下有效”，不能写成全局有效。

### 不允许写的 claim

以下说法当前证据不支持：

- 不能说方法全面 SOTA。
- 不能说所有模块都有正贡献。
- 不能说 DPS/Calibration 在所有 k 下有效。
- 不能说 ordinary similarity-aware replay 已经解决 source-target mismatch。
- 不能说 Gate 有效。
- 不能说异常检测问题已经解决；当前主要证据是 forecasting MAE/RMSE 和鲁棒性，不是 anomaly detection F1/AUPRC。
- 不能把 BDG2 的低 median MAE 解读成模型极强；必须说明 near-zero/inactive buildings 的影响。

### 可以写的 defensible claim

可以写：

```text
Residual RevIN consistently improves persistence-anchored cold-start forecasting
across COFACTOR buildings and adaptation lengths.
```

可以写：

```text
Multi-scale residual encoding further improves the RevIN-based residual model on
both COFACTOR and BDG2, suggesting that cold-start transfer benefits from explicit
short- and daily-scale residual structure.
```

可以写：

```text
Daily-profile source selection and calibration are not universally beneficial.
They help mainly when the target adaptation window is long enough to estimate
reliable profile statistics, such as k=14 or k=30.
```

可以写：

```text
The results support a modular robustness interpretation rather than a universal
winner: RevIN and MSR are core modules, whereas source selection and calibration
should be used conditionally.
```

### 论文主表还缺的证据

当前模块筛选结果足够支持“模块是否有用”，但还不够支持完整论文主张。主表必须补：

| 对照方法 | 为什么必须有 |
|---|---|
| Persistence | 建筑负荷冷启动最强朴素基线 |
| RF-ST | 已知非常强的 source-target 树模型 |
| HistGBDT-ST 或 ExtraTrees-ST | 证明不是只赢弱模型 |
| DLinear-M3 | 代表 target adaptation neural baseline |
| DLinear-SRFT | 代表已有 neural transfer baseline |
| `M3_ANCHOR_REVIN` | 当前内部强 baseline |
| `M9_REVIN_MSR` | 推荐轻量最终方法 |
| `M11_REVIN_MSR_DPS_CAL` | 推荐完整条件增强方法 |

主表不能只放模块消融。否则审稿人会说“你只和自己比”。

### 必须补的分析

1. **BDG2 active/inactive 分层**
   - 原因：BDG2 有 near-zero/inactive buildings；
   - 不分层会导致 median MAE 和 mean/worst MAE 解释不干净；
   - 需要报告 active-only、inactive-only、all buildings。

2. **负迁移率**
   - 定义建议：方法 MAE 高于 Persistence 或高于 `M1/M3` 的比例；
   - 按 building × k 作为配对单位；
   - 这是鲁棒迁移论文的核心证据。

3. **worst-case / tail error**
   - 报告 max MAE、90th percentile MAE；
   - BDG2 上尤其重要，因为少数建筑会崩溃。

4. **paired statistics**
   - Wilcoxon signed-rank 或 paired bootstrap；
   - 单位是 building × k，不把 seed 当独立样本；
   - 主文可简报 p 值，补充材料放完整表。

5. **模块选择规则**
   - 极端冷启动 k=3：优先 `M9_REVIN_MSR`；
   - k=7：`M9` 或 `M11` 需按验证集选择；
   - k=14/30：可考虑 `M11_REVIN_MSR_DPS_CAL`；
   - 如果不想引入按 k 选择，论文主方法建议定为 `M9`，`M11` 作为 enhanced variant。

### 推荐论文定位

不建议定位为：

```text
一种全新的通用时间序列预测模型。
```

建议定位为：

```text
一篇面向冷启动建筑负荷预测的鲁棒迁移实验与方法论文。
```

最合适的标题方向：

```text
Persistence-Anchored Multi-Scale Residual Transfer for Cold-Start Building Load Forecasting
```

或更保守：

```text
Robust Residual Transfer under Persistence-Dominated Cold-Start Building Load Forecasting
```

### 下一步执行顺序

1. 汇总最终主表：`Persistence / RF-ST / HistGBDT-ST / DLinear-M3 / DLinear-SRFT / M3 / M9 / M11`。
2. 做 BDG2 active/inactive 分层表。
3. 做 paired win-rate、negative transfer rate、worst-case MAE 表。
4. 画 2 张主图：
   - Figure 1：问题与方法框架，突出 persistence anchor、RevIN、MSR、DPS；
   - Figure 2：模块消融结果，按 k 展示 `M1/M3/M9/M11`。
5. 写 Results 时按问题组织：
   - RQ1：强 persistence 下，直接迁移为什么不稳；
   - RQ2：RevIN 是否缓解分布漂移；
   - RQ3：MSR 是否带来跨数据集稳定收益；
   - RQ4：DPS/Calibration 什么时候有用，什么时候不该用。
