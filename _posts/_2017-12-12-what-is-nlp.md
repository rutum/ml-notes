---
layout: page
title: What is Natural Language Processing (NLP)?
filter: [blog]
tags: nlp introduction
categories: [nlp introduction]
author: rutum
---

Last year [I wrote a highly popular blog post about Natural Language Processing, Machine Learning, and Deep Learning](http://rutumulkar.com/blog/2016/NLP-ML). 

In this post, we will break down NLP further and talk about Rule-Based and Statistical NLP. I will discuss why everyone needs to know about NLP and AI (Artificial Intelligence), how Machine Learning (ML) fits into the NLP space (it is indispensable actually) and how we are using it in our daily life even without knowing about it. 

#### Introduction to NLP

Natural Language Processing or NLP is a phrase that is formed from 3 components - *natural* - as exists in nature, *language* - that we use to communicate with each other, *processing* - something that is done automatically. Putting these words together, we get *Natural Language Processing* or *NLP* - which stands for the approaches to "process" natural language or human language. 

This is a very generic term. What does this "processing" even mean? 

As a human, I understand English when someone talks to me, is that NLP? Yes! When done automatically, it is called [Natural Language Understanding (NLU)](https://en.wikipedia.org/wiki/Natural_language_understanding). 


| <img src="http://www.stuartduncan.name/wp-content/uploads/2012/05/understanding-1024x834.jpg" alt="Human Understanding. Image Credit: http://www.stuartduncan.name" class="img-responsive center-image" style="width: 400px;"/> |
| Human Understanding. Image Credit: http://www.stuartduncan.name |


I translated some Hindi to English for my friend, is that NLP? Yes, it is called [Machine Translation (MT)](https://en.wikipedia.org/wiki/Machine_translation) when done automatically. 


|<img src="https://cloud.google.com/images/products/artwork/hello-lead.png" class="img-responsive center-image" width="200px"/>|
|Machine Translation. Image Credit: Google Cloud |


As humans, we perform Natural Language Processing pretty well but we are not perfect; misunderstandings are pretty common among humans, and we often interpret the same language differently. So, language processing isn't [deterministic](https://en.wikipedia.org/wiki/Deterministic_system) (which means that the same language doesn't have the same interpretation, unlike math where 1 + 1 is deterministic and always equals 2) and something that might be funny to me, might not be funny to you. 

This inherent [non-deterministic](https://en.wikipedia.org/wiki/Nondeterministic_algorithm) nature of the field of Natural Language Processing makes it an interesting and an [NP-hard problem](https://en.wikipedia.org/wiki/NP-hardness). In this sense, understanding NLP is like creating a new form of intelligence in an artificial manner that can understand how humans understand language; which is why NLP is a subfield of [Artificial Intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence). NLP experts say that if humans don't agree 100% on NLP tasks (like language understanding, or language translation), it isn't possible to model a machine to perform these tasks without some degree of error. Side note - if an NLP consultant ever tells you that they can create a model that is more precise than a human, be very wary of them. More about that in my post about [7 questions to ask before you hire your Data Science consultant]().

#### Rule-Based NLP vs Statistical NLP
NLP separated into two different sets of ideologies and approaches. 

One set of scientists believe that it is impossible to completely do NLP without some inherent background knowledge that we take for granted in our daily lives such as *freezing temperatures cause hypothermia*, *hot coffee will burn my skin* and so on. This set of knowledge is collectively known as *commonsense knowledge* and has brought about the field of *Commonsense Reasoning* and very many [conferences](http://commonsensereasoning.org/) and companies (a special mention to [Cyc](https://en.wikipedia.org/wiki/Cyc)). 

Encoding commonsense knowledge is a very time intensive and manual effort driven process as is considered to be in the space of [Rules-Based NLP](https://en.wikipedia.org/wiki/Rule-based_system). It is hard because commonsense knowledge isn't found in the written text (discourse), and we don't know how many rules we need to create, before the work for encoding knowledge is complete. Here is an example: as humans, we inherently understand the concepts of *death* and you will rarely find documents that describe it by explaining the existence and nonexistence of hydrocarbons. Similarly are the concepts of *moving* and *dancing* which usually do not require any explanation to a human, but a computer model requires the breakdown of *moving* into the *origin*, *destination*, and the concept of not being at the origin after the move has happened. Dancing, on the other hand, is also a type of moving, but it is obviously very different from a traditional move, and requires more explanation because you can move a lot and still end up in your original location, so what is the point of a *dance move*?

Another set of scientists have taken a different ([now deceptively mainstream](https://www.quora.com/Why-are-rule-based-methods-becoming-unpopular-in-NLP-Are-rule-based-methods-still-in-use-If-yes-where-should-I-look-for-them)) approach to NLP. Instead of creating commonsense data that is missing in textual discourse, their idea is to leverage large amounts of already existing data for NLP tasks. This approach is statistical and inductive in nature and the idea is that if we can find enough number of examples of a given problem, we could potentially solve it using the power of [induction](https://en.wikipedia.org/wiki/Mathematical_induction). Statistical NLP makes heavy use of *Machine Learning* for developing models and deriving insights from a labeled text. 

Rule-Based NLP and Statistical NLP use different approaches for solving the same problems. Here are a couple of examples:

##### Parsing:

- Rules-Based uses Linguistic rules and patterns. E.g English has the structure of SVO (Subject Verb Object), Hindi has SOV (Subject Object Verb). 
- Statistical NLP induces linguistic rules from the text (so our models are only as good as our text), along with lots of labeling of the text to predict the most likely parse tree of a new data source. 

|<img src="http://www.cs.cornell.edu/courses/cs2112/2014fa/lectures/lec_parsing/simple-sentence.png" class="img-responsive center-image" width="300px"/>|
|Natural Language Parsing. Image Credit: cs.cornell.edu|


##### Synonym extraction

- Rules-Based approaches use thesaurus and lists. Data sources such as [Wordnet](https://wordnet.princeton.edu/) are very useful for deterministic rule-based approaches
- Statistical NLP approaches use statistics to induce thesaurus based on [how similar words have similar contexts](https://code.google.com/archive/p/word2vec/) 

|<img src="https://camo.githubusercontent.com/bc16ea3ea66a7f696fa17d656288d07c8dcd1254/687474703a2f2f692e696d6775722e636f6d2f4443466367784e2e706e67" class="img-responsive center-image" width="800px"/>|
|*Synonyms using Word2Vec. Image Credit: oscii-lab*|

##### Sentiment Analysis

- Rules-based approaches look for linguistic terms such as "love", and "hate", "like" and "dislike" etc. and deterministically classify text as positive and negative
- Statistical NLP approaches use Machine Learning and do some feature engineering to provide weights to linguistic terms to determine the positive and negative nature of texts. 

Which approach is better? Both the approaches have their advantages. Rules-based approaches mimic the human mind and present highly precise results, however, they are limited by what we provide as rules. Statistical approaches are less precise, but they have a much higher coverage than rules-based systems as they are able to account for cases that are not explicitly specified in the rules. 

Most institutions prefer to use a hybrid approach to NLP, using Rule-Based along with Statistical Systems.  

#### Solving NLP Problems

We use NLP every day when we do a google search. Search engines use [Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) in their backend, which is one of the subfields of NLP. 

NLP had earned its popularity because of several mainstream types of language-based problems - such as [text summarization](https://en.wikipedia.org/wiki/Automatic_summarization), [sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis), [keyword extraction](https://en.wikipedia.org/wiki/Keyword_extraction), [question answering](https://en.wikipedia.org/wiki/Question_answering), [conversational interfaces and chatbots](https://en.wikipedia.org/wiki/Chatbot), [machine translation](https://en.wikipedia.org/wiki/Machine_translation), to name a few. 

#### Conclusion

Today NLP is largely statistical because of the availability of massive amounts of data. We can use tools like [Word2Vec](https://code.google.com/archive/p/word2vec/) to get us similarly occurring concepts, and search engines like [Elastic Search](https://www.elastic.co/) to organize our text to make it searchable. We are able to use off the shelf tools like [Stanford Core NLP](https://stanfordnlp.github.io/CoreNLP/) to parse our data for us and other algorithms like [Latent Dirichlet Allocation (LDA)](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/) to discover clusters and topics in the text. 

As a consumer, we use NLP every day - from your first google search of the day, to your curated daily news articles delivered to you, your online shopping experience and reading reviews, and your conversational assistants such as [OK Google](http://ok-google.io/), [Alexa](https://developer.amazon.com/alexa), and [Siri](https://en.wikipedia.org/wiki/Siri).

NLP is embedded in our everyday lives, and we use it even without realizing it. The latest wave of [conversational interfaces or chatbots](https://en.wikipedia.org/wiki/Chatbot) are adding a human component to conversation, and we are finally blending the 2 approaches to NLP - Rule-Based and Statistical NLP. 

Where do we go from here? I am excited for the future of NLP, and although we are [very far from NLP/AI taking over the world](https://hbr.org/2016/11/what-artificial-intelligence-can-and-cant-do-right-now) I am optimistic about computational power being more ingrained in our lives to make the world a better and easier place for us. 

To learn more about NLP and for additional NLP resources [check out this cool blog post from Algorithmia](https://blog.algorithmia.com/introduction-natural-language-processing-nlp/). 