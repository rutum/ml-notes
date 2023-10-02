---
layout: page
title: The Variations and Optimizations for the Transformer Architecture
tags: transformers deep-learning generative-AI nlp positional-encoding activation-functions layer-norm generative-language-models genAI llm
---

# The Vanilla Transformer 
## Architecture
- Attention and Multi-Head Self-Attention Mechanism
    - explain Key, Value and Query matrices
        - query: As the current focus of attention when being compared to all of the other query preceding inputs. We’ll refer to this role as a query.
        - Key: In its role as a preceding input being compared to the current focus of attenkey tion. We’ll refer to this role as a key.
        - Value: And finally, as a value used to compute the output for the current focus of attention - it is dependent on Query and Key
    To capture these three different roles, transformers introduce weight matrices $$W^Q$$, $$W^K$$, and $$W^V$$. These weights will be used to project each input vector $$x_i$$ into a representation of its role as a key, query, or value.
    $$q_i = W^Qx_i; k_i = W^Kx_i; v_i = W^Vx_i$$
    The inputs $$x$$ and outputs $$y$$ of transformers, as well as the intermediate vectors after
the various layers, all have the same dimensionality $$1×d$$. For now let’s assume the
dimensionalities of the transform matrices are $$W^Q \in R^{d×d}, W^K ∈ R^{d×d}$$, and $$W^V ∈ R^{d×d}$$. Later we’ll need separate dimensions for these matrices when we introduce multi-headed attention,

- explain batch normalization to address vanishing gradients
Layer normalization (or layer norm) is one of many forms of normalization that
can be used to improve training performance in deep neural networks by keeping
the values of a hidden layer in a range that facilitates gradient-based training.
- explain dropout
- explain semantic embedding for a sentence - CLS - how is CLS used? 
- try to run the code if you can

https://github.com/leonardoaraujosantos/Transformers/blob/master/Vanila_Transformer.ipynb

## Limitations of the Vanilla Transformer
### Capturing Long Sequences with Transformers
- it takes quadratic time to work with long sequences
### Positional Encoding and improvements
- what are the limitations of OHE based positional encodings? How are these OHE encodings incorporated into the input of the transformer?
### Activation Function and improvements
- which activation function was used? ReLU? 
### Layer Normalization improvements
- layer norm was done after a single layer, but there are several other ways og doing it. What are the limitations of doing it after FF layer? - how is layer norm different from batch norm?

# Transformers for Semantic Embeddings
- BERT, Roberta, Distilbert, Transformer XL, T5, Backpack models (ACL 2023 https://arxiv.org/pdf/2305.16765.pdf)
- what was the objective function for each one of the above models? 
- can the embeddings be fine tuned for donwstream tasks?

# Transformers for Generative Language Modeling
- GPT1, GPT2, GPT3, GPT4, LLaMA, Palm, BARD

Rotary Position Embeddings (Rope) -https://paperswithcode.com/method/rope
https://huggingface.co/docs/transformers/model_doc/roformer - roformer
https://sh-tsang.medium.com/brief-review-roformer-enhanced-transformer-with-rotary-position-embedding-36f67a619442

# References
Hewitt et. al., [Backpack Language Models](https://arxiv.org/pdf/2302.07730.pdf), ACL, 2023

https://github.com/leonardoaraujosantos/Transformers/blob/master/Vanila_Transformer.ipynb
