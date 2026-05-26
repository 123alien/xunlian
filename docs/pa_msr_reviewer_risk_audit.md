# PA-MSR 论文证据链与审稿风险审计

日期：2026-05-19

## 当前中心 claim

本文不主张“一个 full model 在所有场景全面最好”。当前可支撑的中心 claim 是：

> 在冷启动建筑下一小时负荷预测中，强 persistence 和 source-target tree ensembles 会显著改变迁移学习结论；直接神经迁移存在 worst-case failure 和 negative transfer 风险。将任务改写为 persistence-anchored residual prediction，并加入 residual RevIN 与 multi-scale residual encoding，可以在 BDG2-24 和 COFACTOR-44 上形成更稳健的冷启动预测框架。PA-MSR 是极端冷启动默认版本；PA-MSR+ 是 k=14/30 时的条件增强版本。

## 证据已经能支撑的部分

1. **强基线必要性**
   - BDG2-24 中 Persistence、RF-ST、ExtraTrees-ST、HistGBDT-ST 都很强。
   - DLinear-SRFT 等直接神经迁移出现很大的 worst-case MAE。
   - 可写成 benchmark/evaluation contribution。

2. **PA-MSR 在 BDG2-24 上成立**
   - `M9_REVIN_MSR` 在 k=3/30 mean MAE 第一。
   - `M11_REVIN_MSR_DPS_CAL` 在 k=7/14 mean MAE 第一。
   - PA-MSR vs `M3_ANCHOR_REVIN` 在 BDG2-24 四个 k 均显著，win rate 79.2%-87.5%。
   - PA-MSR/PA-MSR+ 明显优于 RF-ST、ExtraTrees-ST、HistGBDT-ST。

3. **COFACTOR-44 外部验证可用**
   - COFACTOR-44 没有 BDG2 inactive/near-zero 的主要问题。
   - PA-MSR 在 k=3/7 aggregate leaderboard 第一。
   - PA-MSR+ 在 k=14/30 aggregate leaderboard 第一。
   - PA-MSR vs ExtraTrees-ST 的 win rate 为 97.7%、75.0%、65.9%、88.6%。
   - 因此可以写“外部 active-building 数据集上 generalizes”，但要承认 pairwise gains 不均匀。

4. **模块结论可写，但要分层**
   - Core：persistence anchoring、residual RevIN、MSR。
   - Conditional：DPS + calibration。
   - Not supported：ordinary similarity、gate。

## 主要审稿风险

1. **BDG2-24 样本量小**
   - 风险：审稿人会问为什么不是全部 BDG2。
   - 当前应对：BDG2-120 已补 PA-MSR-only 规模验证，可以说 PA-MSR 在 120 栋 active subset 上直接验证。
   - 写作底线：承认 BDG2-120 不是 full neural matrix，也不是 all-eligible BDG2 evaluation。

2. **BDG2 inactive/near-zero 建筑影响 median**
   - 风险：审稿人认为结果被全零建筑污染。
   - 当前应对：必须报告 active/inactive 分层表。
   - 写作底线：不要只报 median MAE；主表必须有 mean、P90/P95、worst-case、paired win rate。

3. **COFACTOR 模块证据不能过度外推**
   - 风险：COFACTOR-44 上 MSR 相对 M3 的 paired gain 在 k=3/7/14 较小。
   - 当前应对：COFACTOR-44 用于外部 leaderboard；模块强证据主要来自 BDG2-24 和 COFACTOR module screen。
   - 写作底线：不要写“MSR 在所有数据集都显著提升”。

4. **PA-MSR+ 不是默认最优**
   - 风险：full model 在 k=3/7 反而不如 PA-MSR，容易被认为模块堆叠无效。
   - 当前应对：把 PA-MSR+ 写成 moderate cold-start conditional extension。
   - 写作底线：不要叫 “complete model” 或 “final full model”；叫 enhanced/conditional variant。

5. **PARBoost 与 PA-MSR 命名混乱**
   - 风险：审稿人分不清早期 residual boosting baseline 和最终方法。
   - 当前应对：主文已经加入 mapping：`M9_REVIN_MSR = PA-MSR`，`M11_REVIN_MSR_DPS_CAL = PA-MSR+`，PARBoost 是 earlier baseline。
   - 写作底线：图表里 PARBoost 只能作为 baseline，不作为 proposed method。

6. **“顶会模块缝合”不能进入论文叙事**
   - 风险：创新性被审稿人认为是机械替换。
   - 当前应对：论文叙事必须从 building cold-start failure mode 出发，而不是从替换模块出发。
   - 写作底线：只讲问题、机制、证据，不讲“拿模块缝合”。

## 当前建议投稿定位

更适合能源建筑方向的实证+方法论文，而不是纯 ML 顶会式模型论文。

可投方向：

- Energy and Buildings：有机会，但必须把 benchmark protocol、强基线、外部验证、局限性写扎实。
- Journal of Building Engineering：更稳，方法创新要求相对 Energy and Buildings 稍低。
- Applied Energy：BDG2-120 PA-MSR-only 结果增强了规模证据，但如果冲击更高要求期刊，仍可能需要真实部署、天气/业务价值或更完整的全矩阵验证。

## 下一步必须补的写作材料

1. 把 `results/final_claim_evidence/` 的最终表格嵌入主文。
2. 生成或重画图：
   - protocol/no leakage 图；
   - BDG2-24 + COFACTOR-44 leaderboard 图；
   - paired delta 图；
   - active/inactive 或 persistence-stratified 图。
3. 写一个严格的 “Evaluation protocol” 小节，强调 target test never used。
4. 写一个 “What is not claimed” 段落，主动排除 anomaly detection、universal full model、full-BDG2 exhaustion。
5. 补真实参考文献，并确保每个 dataset、baseline、RevIN、DLinear、tree ensemble、building energy transfer learning 都有来源。

## 2026-05-19 本轮已完成

1. 主文已经嵌入 compact evidence tables：
   - BDG2-24 main benchmark；
   - BDG2-24 active-target sensitivity；
   - COFACTOR-44 external validation；
   - PA-MSR building-paired robustness。
2. 已新增 PA-MSR 专用出图脚本：`scripts/make_pa_msr_paper_figures.py`。
3. 已生成 PA-MSR 主图与 source data：
   - `figures/pa_msr/fig1_protocol_no_leakage.*`
   - `figures/pa_msr/fig2_main_leaderboard.*`
   - `figures/pa_msr/fig3_pairwise_robustness.*`
   - `figures/pa_msr/fig4_bdg2_active_inactive.*`
4. 已修正主文表图计划，主图路径切换到 `figures/pa_msr/`，旧 PARBoost 图只保留为 supplementary evidence。
5. 已补 RevIN 真实引用，并核对 COFACTOR Scientific Data 与 Zenodo 记录。
6. 已在 Discussion 中加入显式 “what is not claimed” 段落，主动排除 universal full model、full-BDG2 exhaustion 和 anomaly detection claim。

## 仍需完成

1. 对参考文献做一次统一编号与正文引用检查。
2. 生成最终投稿包前，需要把主文中的表格编号、Figure 编号和 Supplementary 编号统一排版。
3. PA-MSR 在更大 BDG2 active subset 上的轻量验证已完成；Limitations 中应改为保留“BDG2-120 is PA-MSR-only and does not repeat the full neural matrix or all eligible BDG2 buildings”。
