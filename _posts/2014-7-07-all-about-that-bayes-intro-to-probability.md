---
layout: page
title: An Introduction to Probability
filter: [blog]
tags: probability introduction
categories: [probability]
author: rutum
last_modified_at: 2021-05-08
---

This post is an introduction to probability theory. Probability theory is the backbone of AI, and the this post attempts to cover these fundamentals, and bring us to [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), which is a simple generative classification algorithm for text classification.

# Random Variables

In this world things keep happening around us. Each event occurring is a Random Variable. A Random Variable is an event, like elections, snow or hail. Random variables have an outcome attached them - the value of which is between 0 and 1. This is the likelihood of that event happening. We hear the outcomes of random variables all the time - There is a 50% chance or precipitation, The Seattle Seahawks have a 90% chance of winning the game.


# Simple Probability

Where do we get these numbers from? From past data.

| **Year**  | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 |
| **Rain**  | Rainy | Dry | Rainy | Rainy | Rainy | Dry | Dry | Rainy |

$$
p(Rain=Rainy) = \frac{\sum(Rain=Rainy)}{\sum(Rain=Rainy) + \sum(Rain=Dry)} = \frac{5}{8}$$

$$p(Rain=Dry) = \frac{\sum(Rain=Dry)}{\sum(Rain=Rainy) + \sum(Rain=Dry)} = \frac{3}{8}
$$

# Probability of 2 Events

What is the probability that event A and Event B happening together? Consider the following table, with data about the *Rain* and *Sun* received by Seattle for the past few years.

| **Year**  | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 |
| **Rain**  | Rainy | Dry | Rainy | Rainy | Rainy | Dry | Dry | Rainy |
| **Sun** | Sunny | Sunny | Sunny | Cloudy | Cloudy | Cloudy | Sunny | Sunny |


Using the above information, can you compute what is the probability that it will be Sunny and Rainy in 2016?

We can get this number easily from the **Joint Distribution**

<table>
<tbody>
<tr>
    <td colspan="2" rowspan="2"></td>
    <td align="center" colspan="2"><b>RAIN</b></td>
</tr>
<tr>
    <td><b>Rainy</b></td>
    <td><b>Dry</b></td></tr>
<tr>
    <td align="center" rowspan="2"><b>SUN</b></td>
    <td><b>Sunny</b></td>
    <td>3/8</td>
    <td>2/8</td>
</tr>
<tr>
    <td><b>Cloudy</b>
    </td><td>2/8</td>
    <td>1/8</td></tr>
</tbody>
</table>

In 3 out of the 8 examples above, it is *Sunny* and *Rainy* at the same time. Similarly, in 1 out of 8 times it is *Cloudy* and it is *Dry*. So we can compute the probability of multiple events happening at the same time using the *Joint Distribution*. If there are more than 2 variables, the table will be of a higher dimension

We can extend this table further include **Marginalization**. *Marginalization* is just a fancy word for adding up all the probabilities in each row, and the probabilities in each column respectively.

<table>
<tbody>
    <tr>
        <td colspan="2" rowspan="2"></td>
        <td align="center" colspan="3"><b>RAIN</b></td>
    </tr>
    <tr>
        <td><b>Rainy</b></td>
        <td><b>Dry</b></td>
        <td><b>Margin</b></td></tr>
    <tr>
        <td align="center" rowspan="3"><b>SUN</b></td>
        <td><b>Sunny</b></td>
        <td>0.375</td>
        <td>0.25</td>
        <td>0.625</td>
    </tr>
    <tr>
        <td><b>Cloudy</b></td>
        <td>0.25</td>
        <td>0.125</td>
        <td>0.375</td>
    </tr>
    <tr>
        <td><b>Margin</b></td>
        <td>0.625</td>
        <td>0.375</td>
        <td>1</td></tr>
</tbody>
</table>

Why are margins helpful? They remove the effects of one of the two events in the table. So, if we want to know the probability that it will rain (irrespective of other events), we can find it from the marginal table as 0.625. From Table 1, we can confirm this by computing all the individual instances that it rains - 5/8 = 0.625

# Conditional Probability

What do we do when one of the outcomes is already given to us? On this new day in 2016, it is very sunny, but what is the probability that it will rain?

$$
P(Rain=Rainy \mid Sun=Sunny)
$$

which is read as - probability that it will rain, given that there is sun.

This is computed in the same way as we compute normal probability, but we will just look at the cases where Sun = Sun from Table 1. There are 5 instances of Sun = Sun in Table 1, and in 3 of those cases Rain = Rain. So the probability of

$$ P(Rain=Rainy \mid Sun=Sunny) = 3/5 = 0.6 $$

We can also compute this from Table 3.
Total probability of Sun = 0.625 (Row 1 Marginal probability).
Probability of Rain and Sun = 0.375

Probability of Rain given Sun = 0.375/0.625 = 0.6

# Difference between conditional and joint probability

Conditional and Joint probability are often mistaken for each other because of the similarity in their naming convention. So what is the difference between: $ P(AB) $ and $ P(A \mid B) $

The first is Joint Probability and the second is Conditional Probability.

Joint probability computes the probability of 2 events happening together. In the case above - what is the probability that Event A and Event B both happen together? We do not know whether either of these events actually happened, and are computing the probability of both of them happening together.


Conditional probability is similar, but with one difference - We already know that one of the events (e.g. Event B) did happen. So we are looking for the probability of Event A, when we know the Event B already happened or that the probability of Event B is 1. This is a subtle but a significantly different way of looking at things.

# Bayes Rule

$$P(A B) = P(A) \, P(B \mid A)$$

$$ P(B A) = P(B) \, P(A \mid B)$$

$$ P(A B) = P(B A) $$

Equating (4) and (5)

$$ P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)} $$

This is the **Bayes Rule**.

Bayes Rule is interesting, and significant, because we can use it to discover the conditional probability of something, using the conditional probability going the other direction. For example: to find the probability $ P(death \mid smoking)$ , we can get this unknown from $ P(smoking \mid death) $, which is much easier to collect data for, as it is easier to find out whether the person who died was a smoker or a non smoker.

---

Lets look at some real examples of probability in action. Consider a prosecutor, who wants to know whether to charge someone with a crime, given the forensic evidence of fingerprints, and town population.

The data we have is the following:

- One person in a town of 100,000 committed a crime. The probability that is he guilty $ P(G) = 0.00001$, where $P(G)$ is the probability of a person being guilty of having committed a crime
- The forensics experts tell us, that if someone commits a crime, then they leave behind fingerprints 99% of the time. $P(F \mid G) = 0.99$, where $P(F \mid G)$ is the probability of fingerprints, given crime is commited
- There are usually 3 people's fingerprints in any given location. So $P(F) = 3 * 0.00001 = 0.00003$. This is because only 1 in 100,000 people could have their fingerprints

We need to compute:

$$P(G \mid F)$$

Using *Bayes Rule* we know that:

$$P(G \mid F) = \frac{P(F \mid G) \, P(G)}{P(F)}$$

Plugging in the values that we already know:

$$P(G \mid F) = \frac{0.99 * 0.00001}{0.00003}$$

$$P(G \mid F) = 0.33$$

This is a good enough probability to get in touch with the suspect, and get his side of the story. However, when the prosecutor talks to the detective, the detective points out that the suspects actually lives at the scrime scene. This makes it highly likely to find the suspect's fingerprints in that location. And the new probability of finding fingerprints becomes : $P(F) = 0.99$

Plugging in those values again into (9), we get:

$$P(G \mid F) = \frac{P(F \mid G) \, P(G)}{P(F)}$$

$$P(G \mid F) = \frac{0.99 * 0.00001}{0.99}$$

$$P(G \mid F) = 0.00001$$

So it completely changes the probability of the suspect being guilty.

This example is interesting because we computed the probability of a $P(G \mid F)$ using the probability of $P(F \mid G)$. This is because we have more data from previous solved crimes about how many peple actually leave fingerprints behind, and the correlation of that with them being guilty.

---

Another motivation for using conditional probability, is that conditional probability in one direction is often less stable that the conditional probability in the other direction. For example, the probability of disease given a symptom $P(D \mid S)$ is less stable as compared to probability of symptom given disease $P(S \mid D)$

So, consider a situation where you think that you might have a horrible disease *Severenitis*. You know that Severenitis is very rare and the probability that someone actually has it is 0.0001. There is a test for it that is reasonably accurate 99%. You go get the test, and it comes back positive. You think, "oh no! I am 99% likely to have the disease". Is this correct? Lets do the Math.

Let $P(H \leftarrow w)$ be the probability of Health being *well*, and $P(H \leftarrow s)$ be the probability of Health being *sick*. Let and $P(T \leftarrow p)$ be the probability of the Test being *positive* and $P(T \leftarrow n)$ be the probability of the Test being *negative*.

We know that the probability you have the disease is low $P(H \leftarrow s) = 0.0001$. We also know that the test is 99% accurate. What does this mean? It means that if you are sick, then the test will accurately predict it by 99%

$P(T \leftarrow n \mid H \leftarrow w) = 0.99$

$P(T \leftarrow n \mid H \leftarrow s) = 0.01$

$P(T \leftarrow p \mid H \leftarrow w) = 0.01$

$P(T \leftarrow p \mid H \leftarrow s) = 0.99$

We need to find out the probability that you are *sick* given that the test is *positive* or $P(H \leftarrow s \mid T \leftarrow p)$

Using Bayes Rule:

$$P(H \leftarrow s \mid T \leftarrow p) = \frac{P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s)}{P(T \leftarrow p)}$$

We know the numerator, but not the denominator. However, it is easy enough to compute the denominator using some clever math!

We know that the total probability of

$P(H \leftarrow s \mid T \leftarrow p) + P(H \leftarrow w \mid T \leftarrow p) = 1$

$$P(H \leftarrow s \mid T \leftarrow p) = \frac{P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s)}{P(T \leftarrow p)}$$

$$P(H \leftarrow w \mid T \leftarrow p) = \frac{P(T \leftarrow p \mid H \leftarrow w) \, P(H \leftarrow w)}{P(T \leftarrow p)}$$

Adding (16) and (17), and equating with (15) we get:

$$\frac{P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s)}{P(T \leftarrow p)} + \frac{P(T \leftarrow p \mid H \leftarrow w) \, P(H \leftarrow w)}{P(T \leftarrow p)} = 1$$

Therefore:

$$ P(T \leftarrow p) = P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s) + P(T \leftarrow p \mid H \leftarrow w) \, P(H \leftarrow w)$$

Substituting (7) into (4) we get:

$$P(H \leftarrow s \mid T \leftarrow p) = \frac{P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s)}{P(T \leftarrow p \mid H \leftarrow s) \, P(H \leftarrow s) + P(T \leftarrow p \mid H \leftarrow w) \, P(H \leftarrow w)}$$

$$P(H \leftarrow s \mid T \leftarrow p) = \frac{0.99 \times 0.0001}{0.99 \times 0.0001  + 0.01 \times 0.9999} $$

$$ = 0.0098$$

This is the reason why doctors are hesitant to order expensive tests if it is unlikely tht you have the disease. Even though the test is accurate, rare diseases are so rare that the very rarity dominates the accuracy of the test.

---

# Naive Bayes

When someone applies *Naive Bayes* to a problem, they are assuming *conditional independence* of all the events. This means:

$$P(ABC... \mid Z) = P(A \mid Z) \, P(B \mid Z) \, P(C \mid Z) \,...$$

When this is plugged into Bayes Rules:

$$P(A \mid BCD...) = \frac{P(BCD...\mid A ) \, P(A)}{P(BCD...)}$$

$$ = \frac{P(B\mid A ) \, P(C \mid A) P(D \mid A) ... P(A)}{P(BCD...)}$$

8. Let $P(BCD...) = \alpha$ which is the normalization constant. Then,

$$ = \alpha \times P(B\mid A ) \, P(C \mid A) P(D \mid A) ... P(A)$$


What we have done here, is assumed that the events A, B, C etc. are not dependent on each other, thereby reducing a very high dimensional table into several low dimensional tables. If we have 100 features, and each feature can take 2 values, then we would have a table of size $2^{100}$. However, assuming independence of events we reduce this to one hundred 4 element tables.

> Naive Bayes is rarely ever true, but it often works because we are not interested in the right probability, but the fact that the correct class has the highest probability.


# Further Reading

1. <a href = "http://www.usna.edu/Users/cs/crabbe/SI420/current/classes/naivebayes/naivebayes.pdf"> A gentle review of Basic Probability</a>


