# 论文文献库 — 总索引

> 最后更新：2026-07-02

---

## 目录结构

```
papers/
├── 01-DQN基础方法/          # DQN算法及其变体
├── 02-多标签分类与RL/       # 多标签分类与强化学习的交叉
├── 03-医疗诊断RL应用/       # RL在医疗诊断中的应用
├── 04-奖励函数设计/          # 奖励函数塑形与设计方法
├── 05-稀有类别与长尾学习/   # 长尾/稀有类别的RL奖励设计
├── 06-假阳性与代价敏感/     # 假阳性控制与代价敏感奖励
├── 07-探索与内在动机/       # 内在动机与探索性奖励
├── 08-中医相关/             # 中医领域相关论文
├── 09-开源框架/             # 代码框架与实现参考
├── 消融实验与对比实验综合分析.md
├── 奖励函数重新设计_参考资料.md
├── 模型改进诊断与路线图.txt
└── 改进资料整理.txt
```

---

## 一、01-DQN基础方法

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 1710.02298_Rainbow_DQN.pdf | Rainbow: Combining Improvements in Deep RL | 2017 | Rainbow DQN 经典论文 |
| 2411.03820_Beyond_The_Rainbow.pdf | Beyond The Rainbow: Extensions | 2024 | Rainbow 扩展 |
| 2512.22186_Curriculum_Dueling_DDQN.pdf | Curriculum Dueling DDQN | 2025 | 课程学习+DDQN |
| 2602.09810_Double_vs_Dueling_DQN_Transfer.pdf | Double vs Dueling DQN Transfer | 2026 | Double/Dueling DQN对比 |

## 二、02-多标签分类与RL

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 1709.04093_Set_Cardinality_State_Distribution.pdf | Set Cardinality State Distribution | 2017 | 多标签集合基数建模 |
| 1809.03118_Deep_Reinforced_Sequence_to_Set.pdf | Deep Reinforced Sequence-to-Set | 2018 | 序列到集合的多标签RL |
| 1904.06690_Deep_Reinforced_Sequence_to_Set_Multi_Label_ACL2019.pdf | Deep Reinforced Sequence-to-Set (ACL) | 2019 | ACL 2019 正式版 |

## 三、03-医疗诊断RL应用

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 2302.10261_Deep_RL_Cost_Effective_Medical_Diagnosis.pdf | Deep RL for Cost-Effective Medical Diagnosis | 2023 | 医疗诊断RL |
| 2304.12828_PrescDRL.pdf | PrescDRL: Prescription Recommendation | 2023 | 处方推荐RL |

## 四、04-奖励函数设计

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 2501.00989_Bootstrapped_Reward_Shaping.pdf | Bootstrapped Reward Shaping | 2025 | 自举奖励塑形 |
| 2502.01307_Improving_PBRS_Effectiveness.pdf | Improving PBRS Effectiveness | 2025 | 奖励塑形改进 |
| 2508.18474_DRTA_Dynamic_Reward_Scaling.pdf | DRTA: Dynamic Reward Scaling | 2025 | 动态奖励缩放（也在06文件夹有副本） |

## 五、05-稀有类别与长尾学习 ✅已下载

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 2510.17520_Curiosity_Meets_Cooperation_Long_Tail_Multi_Label.pdf | Curiosity Meets Cooperation (CD-GTMLL) | 2025 | 好奇心奖励+博弈论 |
| 2602.15330_Scalable_Curiosity_Driven_Game_Theoretic_Long_Tail.pdf | Scalable Curiosity-Driven Framework | 2026 | 可扩展好奇心框架 |
| 2406.16293_SL_RL_Multi_Label_Partial_Labels_Jia2024.pdf | SL+RL for Multi-Label with Partial Labels | 2024 | 局部+全局奖励 |

> ⚠️ Hartvigsen et al. 2020 (KDD) 无法下载，见 `README_未下载文献.md`

## 六、06-假阳性与代价敏感 ✅已下载

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| MEDTRIC_Clinical_Metric_Multi_Label_Diagnosis_Saha2023.pdf | MEDTRIC: Clinical Metric | 2023 | 临床代价敏感评估 |
| 2508.18474_DRTA_Dynamic_Reward_Scaling.pdf | DRTA: Dynamic Reward Scaling | 2025 | 动态奖励缩放 |

> ⚠️ Huang 2025, Araf et al. 2024 无法下载，见 `README_未下载文献.md`

## 七、07-探索与内在动机 ✅已下载

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| Akalin_Loutfi_2021_RL_Approaches_Social_Robotics_Intrinsic_Motivation.pdf | RL in Social Robotics | 2021 | 内在动机奖励综述 |

## 八、08-中医相关

| 文件 | 论文 | 年份 | 说明 |
|------|------|------|------|
| 2502.04345_JingFang_LLM_TCM.pdf | JingFang LLM for TCM | 2025 | 中医经方LLM |

## 九、09-开源框架

| 文件夹 | 说明 |
|--------|------|
| BTR/ | 多标签分类工具包 |
| Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis/ | 医疗诊断RL参考实现 |
| dopamine/ | Google Dopamine RL框架 |
| stable-baselines3/ | Stable-Baselines3 RL框架 |

---

## 十、分析文档

| 文件 | 说明 |
|------|------|
| 消融实验与对比实验综合分析.md | 全部14个模型排名、6个核心发现、论文启示 |
| 奖励函数重新设计_参考资料.md | 10篇文献整理、多目标奖励函数方案、消融实验设计 |
| 模型改进诊断与路线图.txt | 模型问题诊断与改进路线 |
| 改进资料整理.txt | 改进参考资料汇总 |

---

## 十一、未下载文献（需手动获取）

| 论文 | 分类 | 获取方式 |
|------|------|---------|
| Hartvigsen et al. 2020 — RHC KDD | 05 | ACM Digital Library |
| Huang 2025 — RL Classifier | 06 | HAL Science |
| Araf et al. 2024 — Cost-Sensitive Review | 06 | Springer Link |

详细获取方式见各文件夹下的 `README_未下载文献.md`
