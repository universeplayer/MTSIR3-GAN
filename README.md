# MTSIR3-GAN

This repository contains the code and resources for the thesis project "Exploring Generative Adversarial Networks for Multivariate Time Series Data Imputation".

**Core Contribution:**

This project introduces **MTSIR3-GAN**, a novel approach for Multivariate Time Series Imputation (MTSI). It successfully adapts the modern, principled R3GAN architecture [NeurIPS 2024] – originally designed for image generation – to the unique challenges of temporal data.

**Key Features:**

* **Model:** Implements MTSIR3-GAN, leveraging R3GAN's stable training objective (RpGAN + R1 + R2) and modernized convolutional backbone.
* **Innovation:** Features a bespoke time-series patching strategy to enable R3GAN's application to sequential data.
* **Evaluation:** Provides comprehensive experimental results demonstrating MTSIR3-GAN's competitive performance against baselines (TimesNet, SSGAN) on standard benchmark datasets (PhysioNet Challenge 2012, Beijing Air Quality, PSM).
* **Robustness:** Shows effectiveness in handling complex temporal dependencies and robustness towards data anomalies.
* **GUI:** Includes an interactive Dash-based interface for demonstrating the imputation system (details in Chapter 4).

**Goal:**

To address the critical problem of missing data in multivariate time series by developing and validating a stable, high-performing GAN-based imputation method.

**Keywords:**

Multivariate Time Series Imputation (MTSI), Generative Adversarial Networks (GANs), R3GAN, Deep Learning, Time Series Analysis, Missing Data, Data Imputation, PhysioNet, Air Quality, PSM.
