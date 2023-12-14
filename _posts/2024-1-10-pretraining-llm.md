---
layout: page
title: Pretraining of Large Language Models 
tags:  
categories: [generative AI, pretraining, LLMs]
---

# Data Collection and Preprocessing
The scale and quantity of data along with how this data is preprocessed is critical for LLMs to attain powerful capabilities (such as in-context learning, chain-of-thought reasoning etc.) In this section, I will discuss different training data, pretraining methods, data sources, and how pretraining data impacts the performance of LLMs. 

## Data Sources
## Data Pre-processing
## Effect of Pre-training Data on LLMs

# Model Architectures
## Architectures
### Encoder Decoder Architecture
### Causal Decoder Architecture
### Prefix Decoder Architecture
## Detailed Configuration
### Normalization Methods
- LayerNorm
- RMSNorm
- DeepNorm
### Normalization Position
- Post-LN
- Pre-LN
- Sandwich-LN
### Activation Functions
### Positional Embeddings
In the transformer architecture, positional embeddings are employed to inject the absolute or relative position information for modeling sequences. 
- Absolute Position Embeddings
- Relative Position Embedding 
- Rotary Position Embedding ([RoPE](https://arxiv.org/pdf/2310.01924.pdf))
- ALiBi
### Attention and Bias
- Full Attention 
- Sparse Attention
- Multi-Query Attention
- Flash Attention
## Pre-training Tasks
### Language Modeling
### Denoising Autoencoding
### Mixture of Denoisers
## Things to consider while picking an architecture

# Model Training
## Optimization Settings
### Batch Training
### Learning Rate
### Optimizer
### Stabilizing the training
## Scalable Training Architectures
### 3D Parallelism
- Data Parallelism
- Pipeline Parallelism
- Tensor Parallelism
- ZeRO
- Mixed-Precision Training
- Things to consider while training

# Conclusions and Closing Thoughts