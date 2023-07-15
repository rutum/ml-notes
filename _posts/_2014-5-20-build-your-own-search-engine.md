---
layout: page
title: Build your own search Engine
filter: [blog]
tags: information_retrieval lucene
categories: [information retrieval, lucene] 
author: rutum
---

In this post, I will take you through the steps for calculating the $tf \times idf$ values for all the words in a given document. To implement this, we use a small dataset (or corpus, as NLPers like to call it) form the [Project Gutenberg Catalog](http://www.gutenberg.org/). This is just a simple toy example on a very small dataset. In real life we use much larger corpora, and need some more sophisticated tools in order to handle large amounts of data. To brush up on the basic concepts of $tf \times idf$ you might want to check out my post on the [The Math behind Lucene](../core-of-lucene/index.html).

For this exercise, we will use the [Project Gutenberg Selections](http://www.gutenberg.org/) which are released as part of [NLTK Data](http://nltk.googlecode.com/svn/trunk/nltk_data/index.xml). NLTK - Natural Language Toolkit - is a python based module for text processing. The corpus - Project Gutenberg Selections - contains 18 files. Each file is a complete book. Our task is to calculate the $tf \times idf$ of all the words for each of the documents provided. In the end of this exercise we will have 18 documents, with the $tf \times idf$ of each word in each of the documents.

Documents with $tf \times idf$ values for each word (or token) are often used (with the [vector space model](http://en.wikipedia.org/wiki/Vector_space_model)) to compute the similarity between two documents. Such statistics are quite relevant in Information Retrieval, Search Engines, Document Similarity, and so on.

>NOTE: All the materials needed for this exercise (code + data) can be downloaded from my [github repo](https://github.com/rutum/tf-idf).

The code is written in perl, and is heavy in regular expressions. I have tried my best to document the code, but if you have any issues, or discover bugs, please do not hesitate in contacting me. I will not go into the details of all my code in this post, but will highlight a few features.

This is one of the most important tasks of any NLP application. Discourse often contains non-ascii characters, no alphanumeric characters, spaces, and so on. The first and most important task is to remove these unwanted characters from text, to clean it up. Here is a set of regular expressions (regex) that is helpful for this dataset. However it is not an exhaustive set of regexes. For instance, I have not used any regex to convert [UTF8](http://en.wikipedia.org/wiki/UTF-8) to [ASCII](http://en.wikipedia.org/wiki/ASCII).

{% highlight perl %}

def test():
            #remove endline character
            chomp($txt);
            #remove extra space characters
            $txt =~ s/[\h\v]+/ /g;
            #remove caps
            $txt =~ tr/[A-Z]/[a-z]/;
            #remove non-alphanumeric characters
            $txt =~ s/[^a-zA-Z\d\s]//g;

{% endhighlight %}

The rest of my code can be downloaded from my [github repo](https://github.com/rutum/tf-idf). The code is relatively easy to understand if you are familiar with regular expressions, and understand perl. I have provided my solutions in the output directory, within the folder $tf \times idf$. The intermediate $tf$ and $idf$ results are also provided in the output folder.

It is interesting to note here that words occurring in all the documents have an idf value of 0. This means that their tf*idf value will also be 0, deeming them insignificant in contributing to the document vector. These include words like - a, the, when, if, etc. A longer list can be developed by sorting all the values in the file idf.txt (downloaded from my [github repo](https://github.com/rutum/tf-idf)).

The next step, is to use these metrics to compute document similarity. Discovering similar document in a corpus of documents remains one of the most important problems of information retrieval. This is often done by converting documents into document vectors, and comparing the similarity of these vectors to each other using vector similarity metrics.

Note: All the code for this can be downloaded from my [github repo](https://github.com/rutum/document_similarity).

This post uses the output generated in the toy example for $tf \times idf$ which is available in this [github repo](https://github.com/rutum/document_similarity).

CREATING DOCUMENT VECTORS

After computing the $tf \times idf$ values for each document in a given corpus, we need to go through the exercise to convert these values into a document vector. If you recall some vector basics, a vector constitutes a magnitude and a direction. In our case, the direction is represented by a word in the document and magnitude is the weight or the $tf \times idf$ value of the word in the document. In order to simplify our vector calculations we pre-specify the locations of each word in the array representing the document, creating a sparse vector.

For instance consider the vocabulary of the entire corpus to be:
*John, Mary, Susan, Kendra, sang, for, with, plays*

We will assign a location to each word in the vocabulary as:

|John |   Mary |   Susan   |Kendra  |sang   | for |with |   dances|
|0  | 1  | 2 |  3 |  4 |  5 |  6 |  7|

Given this data, for the sentence: John sang for Mary, we create the following boolean document vector:

|John  |  Mary |   Susan |  Kendra | sang |   for |with  |  dances|
|1 |  1  | 0  | 0 |  1 |  1 |  0  | 0|

Here a 1 represents the existence of a word in the text, and a 0 represents the absence of a word from the text. This example is simply to illustrate the concept of creating a document vector. In our toy example we will replace with 1 values with the tf*idf values of a given word in the given document.

Computing the dot Product of two Vectors


 ||John|  Mary | Kendra | sang | for |with | dances | Sentence|
|d1|  0 |  0.27 | 0 | 0 | 0 | 0.1 | 0.1 |John dances with Mary.|
|d2|  0 | 0  | 0.1 |0.27 | 0.27 | 0 | 0 |John sang for Kendra.|
|d3|  0 |  0 |  0.1 |0 | 0 | 0.1 |0.1 |John dances with Kendra.|

dot product between d1 and d2:

$$ d1 \cdot d2 = d1.John * d2.John + d1.Mary * d2.Mary + ....$$

The final similarity scores are:

|doc3.txt |doc2.txt |0.178554901188263|
|doc3.txt |doc1.txt |0.377800203993898|

|doc2.txt |doc3.txt |0.178554901188263|
|doc2.txt |doc1.txt |0|

|doc1.txt |doc3.txt |0.377800203993898|
|doc1.txt |doc2.txt |0|


Notice that document 1 is most similar to document 3. Document 2 and document 1 have absolutely no similarities. This is because both these documents have just one word in common - *John*, which is common to all the documents, so has a weight of 0.

The code is documented, and can be downloaded from my [github repo](https://github.com/rutum/document_similarity).

Happy Coding!
