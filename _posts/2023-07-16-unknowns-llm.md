---
layout: page
title: What we do not know about LLMs
tags: llm ml deep-learning
categories: [llm, ml, deep-learning]
---

Although Deep Learning in general and Transformers in particular have made tremendous strides in applications and downstream tasks (such as Question Answering, Text Summarization, object detection etc.), there are still a lot of gaps in our understanding and effectiveness of Deep Learning and in particular in LLMs. For instance, Generative Large Language Models (LLMs) often hallucinate and produce incorrect content (we don’t know why they do this or how to stop it), Pre-trained Deep Learning Models get outdated and cannot be supplemented with new information (each of them has a checkpoint at which the pre-training stops), Models exhibit biases they they inherit from their training data (we don’t know how to find that bias and mitigate it in the model), and models are still extremely challenging to evaluate. This field is massive, and my research below is a small drop in the bucket as compared to the exponential number of papers that have been published on this topic within the past year. In 2018, before the advent of LLMs as they are today, (Marcus 2018) wrote about the limitations of deep learning, several of which are probably not transferrable to the LLMs of today’s world. There have been several other papers that have highlighted the risks and limitations of LLMs as they exist today - including [the stochastic parrots paper](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) (Bender and at. al. 2021), the foundation models report (Bommasani et. al. , 2021), and [DeepMind’s paper on ethical and social harms](https://arxiv.org/pdf/2112.04359.pdf). 

In this doc, I will focus on the unknowns and risks with LLMs as it pertains to some high level topics, with a few selected papers in each space. These are a small list of limitations and unknowns with LLMs (the latest form of Deep Learning), but there are certainly others - such as reducing data and resource dependencies, performing domain adaptation of a model etc. 

## 1. Disinformation, Hallucinations and Reliability of Models

### 1.1 Disinformation

LLMs can be used for Disinformation - where an entity is creating fake information on purpose to deceive a target population. E.g. The earth is flat, or Moon Landings did not happen. There have been a few threads of work to address Disinformation -(Zellers et. al. 2019) created a model to detect fake news with 92% accuracy.  They trained a model called **Grover** to generate and detect fake news. (Buchanan et al. 2021) highlights how models like GPT-3, together with human input can be very effective in generation of disinformation and how this is a significant problem/threat. This continues to remain an unsolved problem. (Kreps et. al. 2022) discusses the impact of disinformation campaigns, and how humans have a difficult time distinguishing disinformation and real information. OpenAI published a document about the released strategies of their models, and the social impact of LLMs (Solaiman et. al. 2019). (McGuffie and Newhouse, 2020) analyze the risk potentials of GPT-3, and how it can be leverage to generate and propagate extremist ideologies. Creating fake news, videos, images remains one of the major threats of Generative AI, and our lack of understanding of how to address this, control this, or even policy it. 

### 1.2 Hallucinations and Reliability of Models

Hallucinations are situations when a model generates output that is incorrect, or false, but it is not intentionally created by a bad actor, it is because of intrinsic issues in the model. I am grouping hallucinations are dis-information, because it is easy to unintentionally propagate incorrect information, when the information looks reasonably plausible. Anecdotally, while writing this document, I used the opportunity to ask ChatGPT for citations, and it produced plausible looking papers by leading authors in this space “G. Hinton, I. Sutskever” and plausible conference “NeurIPS” in plausible years “2020, 2021”, but on further analysis, these papers did not exist! In (Zhang et. al. 2023), the authors analyze GPT-4 and ChatGPT and create a dataset where both these models hallucinate, and then further explain their hallucinations incorrectly, while simultaneously catch their own hallucinations in a separate session. The authors recommend that exclusively training for fluency in language without focusing on factuality has significant downstream effects. In (Li et. al. 2023), the authors share that ChatGPT hallucinates in 11.4% of the queries, and proposed HaluEval which is a dataset to evaluate hallucinations in language models. In (Azaria and Mitchell, 2023), the authors share their initiatives for predicting hallucinations based on the hidden layers of an LLM. This is one of the first initiatives to map the output of an LLM with respect to hallucinations. Mitigating hallucinations in LLMs is still an unknown problem and there is a lot of work in this area to be accomplished. 

## 2. Biases in Deep Learning

In 2021 (Bender et. al. 2021) describe that despite the large scale of data that GPT-2 like models are trained on, they are still having uneven representation of the population, as internet data largely over-represents younger users from developed countries. As per the paper. - “ GPT-2’s training data is based on Reddit, which according to Pew Internet Research’s 2016 survey, 67% of Reddit users in the US are men, 64% between ages 18 and 29 and only 8.8-15% of Wikipedians are female.”
Not representing minorities such as (trans, queer, neurodivergent people) and filtering bad words could further marginalize certain populations. (e.g., LGBT+). The main takeaway from this paper is that it is crucial to understand and document the composition of the datasets used to train large language models. In 2018 (Rudiner et. al. 2018) presented their paper on the inherent gender biases in language, and a new dataset to measure these biases in trained models. They introduce a new benchmark, WinoBias, for coreference resolution focused on gender bias containing sentences with entities corresponding to people referred by their occupation (e.g. the nurse, the doctor, the carpenter). In 2020 (Gehman et. al 2020) looked into the intensity to which pretrained LMs can be prompted to generate toxic language. The authors have created and released REALTOXICITYPROMPTS, which is a dataset of 100K naturally occurring, sentence-level prompts derived from a large corpus of English web text, paired with toxicity scores from a widely used toxicity classifier. Using REALTOXICITYPROMPTS, the authors have found that pretrained LMs can degenerate into toxic text even from seemingly innocuous prompts. Modern LLMs such as ChatGPT have attempted to address this, but these gender biases still surface as part of context. According to (Snyder 2023) the performance reviews that were written by ChatGPT automatically assumed that a “Bubbly receptionist” was was woman and a “strong construction worker” was a man. Bias arises from text, and identifying it, and figuring out how to address this bias to create robust models is still an ongoing research area, with a lot of unknowns. 

## 3. Explainability

It is challenging to get an explanation for “why” a model chose a specific outcome. Although this was challenging for Machine Learning, it has become even more challenging for Deep learning because of the explosion in the number of parameters of a deep learning model, making it harder to debug the inner workings of it. This is particularly important and imperative for Deep Learning applications in Healthcare, where transperancy is required in the model analysis process in order to improve the model robustness. In 2016, (Monroe and Jurafsky 2016) proposed an approach to remove input units, hidden units and other units selectively from the DL model to infer explanations for how the model decided to come up with a certain outcome. In 2019, (Jain and Wallace 2019) published their paper about “Attention is not Explanation”, in which they argue that model weights do not provide a sufficient or meaningful explanation, as learned feature weights are not correlated with the gradient based measure of feature importances. In 2020, (Weigreffe and Pinter 2020) provided a counter argument to (Jain and Wallace 2019) with their paper “Attention is not not Explanation”. Here they provide counter arguments that explainability is subjective in definition, and model weight indeed provide meaningful explanations. In 2023, (Tursun et. al. 2023) propose analyzing the distribution of contextual information using heatmaps for Computer Vision (CV) based tasks. Given this information, explainability for AI is a highly controversial space. It has serious impact in the legal and healthcare space, where trust in the model (no hallucinations) along with the reasoning behind the output are imperative. Chain-of-Thought prompting (Chung et. al. 2022) has pushed the boundaries on the reasoning capabilities of LLMs, the provide some level of explainability, but we are still far from learning how this can be addressed. 

## 4. Domain Adaptation and incorporating external knowledge

Today, there is no universally established approach to take a pretrained model checkpoint and adapt it to a focused domain (such as healthcare, or retail). There have been several initiatives for fine-tuning small and large language models for custom downstream tasks, but it still remains and open question and deeply studied problem. In 2014, (Yosinski et. al. 2014) studied the transferability of features in deep neural networks. The authors argue that the transferability of features decreases as the distance between the base task and target task increases. In 2022, (Guo and Yu, 2022) surveyed large language models to  study the domain adaptation capability of them, and presented new requirements that we need to meet to support them. Some researchers such as (Singh et. al. 2019) have studied domain adaptation by proposing new datasets such as GLUE for Natural Language Understanding Tasks. In this paper they introduce the General Language Understanding Evaluation (GLUE) benchmark, a collection of tools for evaluating the performance of models across a diverse set of existing NLU tasks. 

## 5. Adversarial Attacks due to prompting

In recent literature, it has been observed that certain prompts can enable unpredictable behavior in LLMs. Recently, [ChatGPT seemed to break with some reddit usernames](https://www.vice.com/en/article/epzyva/ai-chatgpt-tokens-words-break-reddit) that was probably present in the training data.  In 2020, (Linyang et. al. 2020) the authors introduce Bert Attack that can use a fine tuned version of BERT to mislead BERT and target BERT models to predict incorrectly. Their code, which is available openly, helps create high quality samples of data for adversarial attacks. In 2023, (Wang et. al. 2023) introduce an ICL (in-Context Learning) attack based on TextAttack to show that adversarial attacks are possible without changing the input to mislead the models. They emphasize the security risks associated with ICL. This area is still under heavy research, to learn about how and why these adversarial attacks can be mitigated. 

## 6. Evaluation of Deep Learning Models

Deep Learning Models are exceptionally challenging to evaluate. The best approaches to evaluate them is using human evaluation, which can be rather expensive. Perplexity (or branching factor) is considered a good way to compare models with each other. [Larger models such as GPT have a lower perplexity as compared to smaller models such as BERT](https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word). Generative models that are decoder only are evaluated using prompting. 

Below are some datasets and their corresponding performance by some LLMs. This is still an evolving field, and new datasets are produced frequently to address emergent capabilities as they are discovered from LLMs. 

In LAMBADA ([Paperno et al. 2016](https://arxiv.org/pdf/1606.06031.pdf)), the task is to predict the last word of a sentence, motivated by the fact that this requires to solve for long-range dependencies.GPT-3 does *much better* on this task than the previous state-of-the-art (based on GPT-2). [[leaderboard](https://paperswithcode.com/sota/language-modelling-on-lambada)]. In HellaSwag ([Zellers et al. 2019](https://arxiv.org/pdf/1905.07830.pdf)), the task is to choose the most appropriate completion for a sentence from a list of choices, to evaluate the commonsense reasoning ability. GPT-3 got close but did not exceed the state-of-the-art. In TriviaQA ([Joshi et al. 2017](https://arxiv.org/pdf/1705.03551.pdf)), the task was to generate a trivia answer given a question, to evaluate for open book reading comprehension and closed book question answering. 

Some other benchmark data include: 
[SWORDS](https://arxiv.org/pdf/2106.04102.pdf): lexical substitution, where the goal is to predict synonyms in the context of a sentence.
[Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300.pdf): 57 multiple-choice problems spanning mathematics, US history, computer science, law, etc.
[TruthfulQA](https://arxiv.org/pdf/2109.07958.pdf): question answering dataset that humans would answer falsely due to misconceptions.

This is a constantly evolving field, and the best approaches for evaluating LLMs (which are generative and are evaluated only using some type of prompting) are still not established. 

## 7. Continued Learning after pre-training on large datasets

One of the ongoing limitations and unknowns with LLMs, is the inability to have continuous training. We need to typically stop training at a checkpoint because of several limitations (model dilution, environmental constraints, data limitations and compute limitations). This also means that models (such as ChatGPT) get outdated based on the model checkpoint.  In (Gururangan et. al. 2020) the authors explain that a second level of pre-training on a domain specific task (domain-adaptive pre-training) leads to performance gains in both high and low resource settings. (Reed et. al. 2021) show a similar application of multiple levels of pre-training, but in computer vision. 

## Conclusion

Deep Learning has come a long way in the past few years, but there are several unknowns in this space that we need to continue to deep dive and learn. 

## **References:**

(Azaria and Mitchell, 2023) A Azaria, T Mitchell, [The internal state of an **llm** knows when its lying](https://arxiv.org/abs/2304.13734),  - arXiv preprint arXiv:2304.13734, 2023 - [arxiv.org](http://arxiv.org/)

(Bender et. al. 2021) Bender, E. M., & Gebru, T., A. McMillan-Major, S. Shmitchell (2021). [On the dangers of stochastic parrots: Can language models be too big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT), Online.

(Bommasani et. al. , 2021) [On the Opportunities and Risks of Foundation Models](https://arxiv.org/pdf/2108.07258.pdf). *Rishi Bommasani, Drew A. Hudson, E. Adeli, R. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, E. Brynjolfsson, S. Buch, D. Card, Rodrigo Castellon, Niladri S. Chatterji, Annie Chen, Kathleen Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, S. Ermon, J. Etchemendy, Kawin Ethayarajh, L. Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, S. Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, G. Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, M. Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, J. Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, A. Narayan, D. Narayanan, Benjamin Newman, Allen Nie, Juan Carlos Niebles, H. Nilforoshan, J. Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, J. Park, C. Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Robert Reich, Hongyu Ren, Frieda Rong, Yusuf H. Roohani, Camilo Ruiz, Jackson K. Ryan, Christopher R’e, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, K. Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramèr, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, M. Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, Percy Liang*. 2021.

(Buchanan et. al. 2021) Ben Buchanan, Andrew Lohn, Micah Musser, Katerina Sedova [Truth, Lies, and Automation](https://cset.georgetown.edu/wp-content/uploads/CSET-Truth-Lies-and-Automation.pdf). *.* CSET report, 2021.

(Chung et. al. 2022) Hyung Won Chung and Le Hou and Shayne Longpre and Barret Zoph and Yi Tay and William Fedus and Yunxuan Li and Xuezhi Wang and Mostafa Dehghani and Siddhartha Brahma and Albert Webson and Shixiang Shane Gu and Zhuyun Dai and Mirac Suzgun and Xinyun Chen and Aakanksha Chowdhery and Alex Castro-Ros and Marie Pellat and Kevin Robinson and Dasha Valter and Sharan Narang and Gaurav Mishra and Adams Yu and Vincent Zhao and Yanping Huang and Andrew Dai and Hongkun Yu and Slav Petrov and Ed H. Chi and Jeff Dean and Jacob Devlin and Adam Roberts and Denny Zhou and Quoc V. Le and Jason Wei, [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf), 2022, arXiv

(Gehman et. al. 2020) Gehman, J., Gururangan, S., Sap, M., Choi, Y., Smith, N. A., & Yarowsky, D. (2020). [RealToxicityPrompts: Evaluating neural toxic degeneration in language models](https://arxiv.org/pdf/2009.11462.pdf). In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Online.

(Guo and Yu, 2022), Xu Guo, and Han Yu, [On the Domain Adaptation and Generalization of Pretrained Language Models: A Survey](https://arxiv.org/pdf/2211.03154.pdf), 2022

(Gururangan et. al. 2020) Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. [Don’t stop pretraining: adapt language models to domains and tasks](https://arxiv.org/pdf/2004.10964.pdf). arXiv preprint arXiv:2004.10964, 2020.

(Jain and Wallace 2019) Jain, Sarthak  and Wallace, Byron C., [Attention is not Explanation](https://aclanthology.org/N19-1357.pdf) (Jain & Wallace, NAACL 2019)

(Kreps et. al. 2022) *Sarah Kreps, R. Miles McCain, Miles Brundage.* Journal of Experimental Political Science,  [All the News That’s Fit to Fabricate: AI-Generated Text as a Tool of Media Misinformation](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/40F27F0661B839FA47375F538C19FA59/S2052263020000378a.pdf/all-the-news-thats-fit-to-fabricate-ai-generated-text-as-a-tool-of-media-misinformation.pdf). , 2020, Journal of Experimental Political Science , [Volume 9](https://www.cambridge.org/core/journals/journal-of-experimental-political-science/volume/EF5AABBCCAC78B8137852EAB223EDB0A), [Issue 1](https://www.cambridge.org/core/journals/journal-of-experimental-political-science/issue/CF103A8FE1E9300A31099A1406540785), Spring 2022 , pp. 104 - 117, DOI: https://doi.org/10.1017/XPS.2020.37

(Li et. al. 2023) J Li, X Cheng, WX Zhao, JY Nie, JR Wen, [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747), - arXiv preprint arXiv:2305.11747, 2023 - [arxiv.org](http://arxiv.org/)

(Linyang et. al. 2020) Li, Linyang et al. “[BERT-ATTACK: Adversarial Attack against BERT Using BERT](https://arxiv.org/pdf/2004.09984.pdf).” *ArXiv* abs/2004.09984 (2020): n. pag.

(Marcus 2018) Gary Marcus, [Deep Learning: A Critical Appraisal](https://arxiv.org/pdf/1801.00631.pdf), , 2018, CoRR, 

(McGuffie and Newhouse, 2020) Kris McGuffie, Alex Newhouse [The Radicalization Risks of GPT-3 and Advanced Neural Language Models](https://arxiv.org/pdf/2009.06807.pdf). . 2020.

(Monroe and Jurafsky 2016) Li, J., Monroe, W., & Jurafsky, D. (2016). [Understanding neural networks through representation erasure](https://arxiv.org/pdf/1612.08220.pdf). arXiv preprint arXiv:1612.08220.

(Reed et. al. 2021) Colorado J. Reed, Xiangyu Yue, Ani Nrusimha, Sayna Ebrahimi, Vivek Vijaykumar, Richard Mao, Bo Li, Shanghang Zhang, Devin Guillory, Sean Metzger, Kurt Keutzer, and Trevor Darrell. [Self-supervised pretraining improves self-supervised pretraining](https://arxiv.org/pdf/2103.12718.pdf). arXiv, 2021.

(Rudiner et. al. 2018) Rudinger, R., Naradowsky, J., Leonard, B., & Van Durme, B. (2018). Gender bias in coreference resolution: [Evaluation and debiasing methods](https://aclanthology.org/N18-2003.pdf). In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), New Orleans, USA.

(Yosinski et. al. 2014) Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). [How transferable are features in deep neural networks?](https://arxiv.org/pdf/1411.1792.pdf) In Advances in Neural Information Processing Systems (NeurIPS), Montreal, Canada.

(Singh et al 2019) Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2019). [GLUE: A Multi-task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/pdf/1804.07461.pdf). In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China.

(Snyder 2023) Kieran Snyder, [We asked ChatGPT to write performance reviews and they are wildly sexist (and racist)](https://www.fastcompany.com/90844066/chatgpt-write-performance-reviews-sexist-and-racist), FastCompany, 2023

(Solaiman et. al. 2019) Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, Jasmine Wang [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/pdf/1908.09203.pdf). . 2019.

(Tursun et. al 2023), Osman Tursun, Simon Denman, Sridha Sridharan, Clinton Fookes, [Towards Self-Explainability of Deep Neural Networks with Heatmap Captioning and Large-Language Models](https://arxiv.org/pdf/2304.02202.pdf), 2023, arXiv

(Wang et. al. 2023) Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, Chaowei Xiao, [Adversarial Demonstration Attacks on Large Language Models](https://arxiv.org/pdf/2305.14950.pdf), 2023, Arxiv, Work in Progress

(Weigreffe and Pinter 2020) Wiegreffe, S., & Pinter, Y. (2020). [Attention is not not Explanation](https://arxiv.org/pdf/1908.04626.pdf). arXiv preprint arXiv:1902.10186.

(Zellers et. al. 2019) Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, Yejin Choi. [Defending Against Neural Fake News](https://arxiv.org/pdf/1905.12616.pdf). NeurIPS 2019. 

(Zhang et. al. 2023) Muru Zhang and Ofir Press and William Merrill and Alisa Liu and Noah A. Smith, [How Language Model Hallucinations Can Snowball](https://arxiv.org/pdf/2305.13534.pdf), 2023, 2305.13534, arXiv




