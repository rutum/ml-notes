---
layout: page
title: What is Byte-Pair Encoding for Tokenization?
filter: [blog]
tags: tokenization
categories: [tokenization]
author: rutum
---

Tokenization is the concept of dividing text into tokens - words (unigrams), or groups of words (n-grams) or even characters. 
Morphology traditionally defines morphemes as the smallest semantic unit. e.g. The word **Unfortunately** can be broken down as **un - fortun - ate - ly**

[[un [[fortun(e) ]$$_{ROOT}$$ ate]$$_{STEM}$$]$$_{STEM}$$ ly]$$_{WORD}$$

Morphology is little studied with deep learning, but Byte Pair Encoding is a way to infer morphology from text. Byte-pair encoding allows us to define tokens automatically from data, instead of precpecifying character or word boundaries. This is especially useful in dealing with unkown words. 

## Modern Tokenizers

Modern tokenizers often automatically induce tokens that include tokens smaller than words - called **Subwords**. E.g. the subwords "-ly", "-ing" give us an ideal about the type of the word - which is what subword tokenization aims to do. 

Most tokenizers have two parts: 
1. **A token learner:** takes a raw training corpus and indices a vocabulary - a set of tokens.
2. **A token segmenter:** takes a raw test sentence and segments it into the tokens in the vocabulary. 

Three algorithms are widely used : 
1. Byte Pair Encoding (Sennrick et. al 2016)
2. Unigram Language Modeling (Kudo 2018)
3. Wordpiece (Schuster and Nakajima 2012) and Sentencepiece (Kudo and Richardson, 2018)

Byte Pair Encoding (BPE) is the simplest of the three. 

## Byte Pair Encoding (BPE) Algorithm

BPE runs within word boundaries. ***BPE Token Learning*** begins with a vocabulary that is just the set of individual characters (tokens). It then runs over a training corpus 'k' times and each time, it merges 2 tokens that occur the most frequently in text. e.g. 'e' and 'r' are merged into a single token 'er' when they occur together in the same order. 

At the end of 'k' iterations, the algorithm produces a list of most frequent 'k' tokens along with the original set of characters. 

{% highlight python linenos %}

def byte_pair_encoding(string_list: List[str], k: int) -> vocab: List[str]:
	 vocab = <list of unique characters in string_list>
	 for i in range(0, k+1):
	 	c_left, c_right = most frequent pair of adjacent tokens in string_list
	 	c_new = c_left + c_right # create a new bigram
	 	vocab = vocab + c_new # add the bigram to teh vocabulary
	 	replace each occurence of c_left, c_right with c_new # update the corpus
	 return(vocab)

{% endhighlight %}



Once the **token learner** learns the vocabulary, the **token parser** is used to tokenize a test sentence from teh learned tokens that were leraned from teh training data. 

In real applications of BPE algorithms BPE is run with many thousands of merges such that most words are represented as tokens and only the rare words are represented by their parts. 

Byte-Pair Encoding was originally a compression algorithm where we replace the most frequent byte pair with a new byte - thereby compressing the data. 

For further reading check out this [NLP class slides from Stanford](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture12-subwords.pdf) or this [chapter on Text Normalization from the Jurafsky and Martin Textbook](https://web.stanford.edu/~jurafsky/slp3/2.pdf)
## References
- T. Kudo. Subword Regularization: [Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/pdf/1804.10959.pdf). 2018
- T. Kudo and J. Richardson. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226.pdf). 2018
- M. Schuster and K. Nakajima. [Japanese and Korea Voice Search](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf). 2012
- R. Sennrich, B. Haddow and A. Birch. [Neural Machine Translation of Rare Words with Subword Units](http://aclweb.org/anthology/P16-1162). ACL 2016

