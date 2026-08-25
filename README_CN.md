# 对抗式精修何时有用？—— R3GAN 用于时间序列填补

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 中文

本仓库是以下论文的代码与论文配套仓：

> **When Does Adversarial Refinement Help? A Negative Result and Open Problem in Adapting R3GAN to Time Series Imputation**
> 何宇峰（香港大学）
> *[第 12 届 SIGKDD 时间序列挖掘与学习研讨会（**MiLeTS 2026**）](https://kdd-milets.github.io/milets2026/)。*
> 📄 论文：[`FMGAN/paper/main_milets2026_sigconf.pdf`](FMGAN/paper/main_milets2026_sigconf.pdf) · [研讨会官方版本](https://kdd-milets.github.io/milets2026/accepted%20papers/16main_milets2026_sigconf%201.pdf)

---

## 一句话总结

扩散模型与 Transformer 已在很大程度上取代 GAN 用于多变量时间序列填补。[R3GAN](https://github.com/brownvc/R3GAN)（NeurIPS 2024）提供了一个现代的正则化相对论 GAN baseline，因此我们问：*将它适配到时序后，能否重新让对抗式精修变得有用？*

我们把 R3GAN 适配到一维时序（**R3GAN-1D**），做成带频域判别器的「粗到精」精修器，并审计了 3 个数据集上的全部 **14 个已保存配置**。每个配置只有一次未记录随机种子的运行，且配置异质，因此这是描述性现象，而不是受控因果消融。**结果是一个边界清晰的负结果：**

- ✅ 5 个 zero/mean-start 已保存配置的 MAE 全部降低，范围为 **48.4–70.2%**。
- ❌ 结果中共有 9 个 linear-start 配置。排除 1 个有明确日志异常的旧运行后，8 个可纳入聚合的配置平均变化为 **−0.7%**，范围 **−3.0% 到 +1.1%**，没有一致的已保存运行收益。
- 🔍 被排除的 AirQuality 旧运行变化为 **−21.9%**；它配置了非零重建权重，但 200 个 epoch 的日志中重建损失始终为零。该运行仍保留展示，但不进入摘要聚合。
- 🧪 当前已保存运行无法识别 R3GAN-1D 的哪个组件造成该现象。决定性的下一步是使用共享 mask、仅在训练集拟合的 scaler 和多个已记录随机种子，做匹配的 reconstruction-only 对照。

这是一篇负结果 workshop 论文：价值在于展示一个基线强度相关的已保存运行现象、公开所有运行与异常，并定义检验判别器是否在生成器重建路径之外增加价值所需的对照。

## 核心结果——精修只救得动弱粗填补

MAE ↓ 越低越好；`Δ` = 相对 MAE 降幅（正=改进）。

| 数据集 | 粗填补方法 | 精修前 | 精修后 | Δ |
|---|---|:---:|:---:|:---:|
| Weather | Zero fill | 0.728 | 0.228 | **+68.6%** |
| | Mean fill | 0.728 | 0.223 | **+69.4%** |
| | 线性插值 | 0.067 | 0.067 | +1.1% |
| Electricity | Zero fill | 0.832 | 0.426 | **+48.7%** |
| | Mean fill | 0.831 | 0.429 | **+48.4%** |
| | 线性插值 | 0.164 | 0.165 | −0.7% |
| AirQuality | Zero fill | 0.765 | 0.228 | **+70.2%** |
| | 线性插值 | 0.151 | 0.152 | −0.4% |

与成熟方法对比（Weather，25% 点缺失）：BRITS **0.039** / SAITS 0.062 / 线性插值 0.067 / R3GAN-1D+线性 0.067 / R3GAN-1D 独立 0.228。

> 上表只展示具有绝对 MAE 的代表性配置。全部 14 个已保存配置是异质的单次运行，不是重复随机种子或受控的粗填补方法消融。大幅降低来自对 trivial baseline 的诊断性比较，不应解读为竞争性性能。

## 复现论文表格

逐次实验的原始输出在 [`FMGAN/results/`](FMGAN/results/)，上表由一个脚本一键重建（无需 GPU、无需训练）：

```bash
python3 FMGAN/analysis.py
```

该脚本会显示全部 14 个运行，明确标记它们是否进入摘要聚合，并在已保存的 `Δ%` 不能由精修前后 MAE 重算时直接报错。其中 8 个可纳入聚合的 linear-start 运行平均为 **−0.7%**（范围 **−3.0% 到 +1.1%**）；旧的 **−21.9%** 日志异常仍在详表中显示，但由脚本内明示的具名排除注册表从摘要聚合中剔除。

## 仓库结构

论文代码在 [`FMGAN/`](FMGAN/)（**F**oundation-**M**odel 粗填补 + **GAN** 精修）：`models/`（R3GAN-1D 架构 + 频域判别器）、`train_refiner.py`（粗到精训练）、`foundation_model/`（MOMENT 包装）、`evaluation/`（指标 + BRITS/SAITS/CSDI baseline）、`data/`（统一加载，含 point/block/subsequence 缺失）、`scripts/`、`results/`（原始 results.json）、`analysis.py`、`paper/`（MiLeTS 2026 camera-ready）。

> 作者早期的**本科毕业设计（FYP）**代码（R3GAN 图像式 MTSI 适配 + SSGAN/TimesNet baseline + Dash GUI）**不属于本论文**，已保留在 [`fyp-archive`](https://github.com/he-yufeng/adversarial-refinement-imputation/tree/fyp-archive) 分支。

## 安装

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

复现表格（`analysis.py`）仅需 Python 标准库；重跑实验需完整栈（PyTorch、PyPOTS、MOMENT）+ GPU，见 [`FMGAN/requirements.txt`](FMGAN/requirements.txt)。

## 局限与开放问题

- 最直接的对照——**recon-only（移除判别器）消融**——是最有价值的下一个实验；当前证据（重建权重 sweep）是间接的。
- **协议内扩散 baseline**（CSDI/FGTI 跑全 3 数据集）、**多 seed 误差棒**、**block/高缺失率/MNAR** 会进一步加强结论。
- 开放问题：在数据处理和模型完全匹配的对照中，判别器能否在同一生成器的重建路径之外提供有用的条件精修信号？当前已保存运行无法因果回答这个问题。

## 引用

见 [README.md](README.md) 的 BibTeX。

## 联系

何宇峰 — [@he-yufeng](https://github.com/he-yufeng) · he-yufeng@connect.hku.hk
