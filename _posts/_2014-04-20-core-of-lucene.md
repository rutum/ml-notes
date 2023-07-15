---
layout: page
title: The Math behind Lucene
filter: [blog]
author: rutum
tags: lucene information_retrieval
categories: [information retrieval, search, lucene]
---
[Lucene](https://lucene.apache.org/) is an open source search engine, that one can use on top of custom data and create your own search engine - like your own personal google. In this post, we will go over the basic math behind Lucene, and how it ranks documents to the input search query.

THE BASICS - TF*IDF

The analysis of language often brings us in situations where we are required to determine the weight or importance of a given word in a document, to determine the relative importance or similarity of a document to another document. In situations such as this, the first step to remove [stop words](http://en.wikipedia.org/wiki/Stop_words) which are basically words that dont contribute to the general focus of a given article. Most common stop words are words like - *a*, *when*, *who*, *what*. The list of stop words keeps changing based on the domain of discourse. For instance, in a corpus of articles about the human heart, the word heart could potentially be a stop word due to the sheer frequency in which it is mentioned. It is always a good idea to remove stop words in a given text before processing it. However, once these stop words are removed, one still faces the task of determining the relative importance or weights of the remaining words in the document - lets start talking about [tf-idf](http://en.wikipedia.org/wiki/Tf*idf).

Observing the words in a document, an intuitive way to discover the importance of a given word is to count the frequency of the word in the document. This is the Term Frequency or the tf of the word in the given document. tf is often normalized so as to not introduce a bias because of the document size. A normalized tf for a given word in a given document is calculated as:

$$tf_{w,d} = \frac{C_{w,d}}{\sum C_{w,d}}$$

where, $$C_{w,d}$$ is the word count of word $$w$$ in document $$d$$ and $\sum C_{w,d}$ is the total count of all words $$w$$ in document $d$.

However, $tf$ by itself is not enough to capture the importance of a word in a document, as there are numerous words like *a*, *the*, *with*, *however*, that have a very high frequency, but less importance. So we need to complement this  with the *Document Frequency* of each word - or $df$. Document Frequency is the total number of documents a given word occurs in. If a word occurs in a large number of documents, it has a high $df$. When this $df$ is used to scale the weight of a word in the document, it is called $idf$ or *Inverse Document Frequency* of that given word.


If $N$ is the total number of documents in a given corpus, then the $IDF$ of a given word in the corpus is:

$$idf_w = log \frac{N}{df_w}$$

Notice that $tf$ is calculated for a given word in a given document, $IDF$ is calculated for a given word over all the documents.

Once we have the $tf$ and $idf$, we can calculate the $tf \times idf$ score for each word to determine the weight of a word in the given document.

The score $tf_w \times idf_{w,d}$ assigns each word $w$ a weight in document. Here are some insights about it.
<ul>
<li>$tf_w \times idf_{w,d}$ is highest when w occurs many times within a small number of documents. This means that the word is significant, and can help establish relevance within the small number of documents.</li>
<li>$tf_w \times idf_{w,d}$ is lower when the word occurs fewer times in a document, or occurs in many documents. If a word occurs infrequently, it might not be of significance. Similarly if a word occurs in a very large number of documents, it probably will not help in discriminating between the document.</li>
<li>$tf_w \times idf_{w,d}$ is lowest when the term occurs in virtually all documents. This covers words like *a*, *an* *the*, *when* etc. which span a large number of documents and have no contribution towards the semantic composition of the document.</li>
</ul>

FROM TF*IDF TO DOCUMENT SIMILARITY

Document vectors are created by computing the relative weights of each word in the document. One way to accomplish this is by computing the $tf \times idf$ values of each of the words in the document. When we compute the $tf \times idf$ of each of the words in a document, we end up with documents with a list of features (words) with their values (weights). In a sense, this represents the document vector. The representation of documents as vectors in a common vector space is known as a *vector space model*, and is the basis of a large number of information retrieval tasks.

A standard way of computing the document similarity is to compute the cosine similarity of the vector representations of the documents. If $d_1$ and $d_2$ are two documents, and $V(d_1)$ and $V(d_2)$ are the vector representations of them respectively, then the similarity of $d_1$ and $d_2$ can be measured as the cosine of the angle between $V(d_1)$ and $V(d_2)$

$$sim(d_1, d_2) = \frac{V(d_1) \cdot V(d_2)}{\mid V(d_1) \mid \mid V(d_2) \mid}$$

In this equation, the numerator is the dot product of vectors $V(d1)$ and $V(d2)$, and the denominator is the product of the *Euclidean length*. Euclidean Length is the sum of squares of the magnitude of each element of the vector.

If $V(d_1)$ and $V(d_2)$ are the following,

$$V(d_1) = [a_1, a_2, a_3, a_4 ...]$$

$$V(d_2) = [b_1, b_2, b_3, b_4 ...]$$

The dot product of $V(d_1)$ and $V(d_2)$ is:

$$V(d_1) . V(d_2) = a_1b_1 + a_2b_2 + a_3b_3 + a_4b_4 + ...$$

The Euclidean length of $V(d_1)$:

$$ \sqrt{a_1^2 +  a_2^2 + a_3^2 + a_4^2 + ... }$$

Similarly, the Euclidean length of $V(d_2)$:

$$ \sqrt{b_1^2 +  b_2^2 + b_3^2 + b_4^2 + ... }$$

Applying (6), (7) and (8) to (3) we get:

$$sim(d_1, d_2) = \frac{a_1b_1 + a_2b_2 + a_3b_3 + a_4b_4 + ...)}{\sqrt{a_1^2 +  a_2^2 + a_3^2 + a_4^2 + ... } \sqrt{b_1^2 +  b_2^2 + b_3^2 + b_4^2 + ... }}$$

Ok, we have cosine similarity. Now what?
What Cosine similarity tells us, is how similar 2 documents are. If the documents are very similar, we have the similarity score closer to 1, but if the documents are completely different, the similarity score is closer to -1.

This is the heart of the scoring mechanism that Lucene uses to retrieve similar documents given a search query. Although Lucene does use a couple of other (mostly user defined) constants to fine tune the results, $tf \times idf$ is the heart of how Lucene operates.

**References:**

1. Introduction to Information Retrieval, by Christopher D. Manning, Prabhakar Raghavan & Hinrich Schütze, [Chapter on Scoring, term weighting and the vector space model](http://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html)
2. [Information Retrieval Facility](http://www.ir-facility.org/scoring-and-ranking-techniques-tf-idf-term-weighting-and-cosine-similarity)
