---
layout: page
title: Understanding your Data - Basic Statistics
filter: [blog]
tags: statistics
categories: [statistics]
author: rutum
---


Have you ever had to deal with a lot of data, and don't know where to start? If yes, then this post is for you. In this post I will try to guide you through some basic approaches and operations you can perform to analyze your data, make some basic sense of it, and decide on your approach for deeper analysis of it. I will use [python](https://www.python.org/downloads/) and a [small subset of  data](https://raw.githubusercontent.com/rutum/basic-statistics/master/bikesharing.csv) from the [Kaggle Bikesharing Challenge](https://www.kaggle.com/c/bike-sharing-demand) to illustrate my examples. The code for this work can be found at [this location](https://github.com/rutum/basic-statistics). Please take a minute to download [python](https://www.python.org/downloads/) and the [sample data](https://raw.githubusercontent.com/rutum/basic-statistics/master/bikesharing.csv) before we proceed.

Below are the topics that I discuss in this post:
- [DESCRIPTION OF DATASET](#description-of-dataset)
- [MIN MAX AND MEAN](#min-max-and-mean)
- [VARIABILITY IN DATA](#variability-in-data)
- [VARIANCE](#variance)
- [STANDARD DEVIATION](#standard-deviation)
- [STANDARD ERROR OF THE MEAN (SEM)](#standard-error-of-the-mean-sem)

## DESCRIPTION OF DATASET

The data provided is a CSV file [bikesharing.csv](https://raw.githubusercontent.com/rutum/basic-statistics/master/bikesharing.csv), with 5 columns - datetime, season, holiday, workingday, and count.

- datetime: The date and time when the statistics were captured
- season: 1 = spring, 2 = summer, 3 = fall, 4 = winter
- holiday: whether the day is considered a holiday
- workingday: whether the day is neither a weekend nor holiday
- count: the total number of bikes rented on that day

## MIN MAX AND MEAN

One of the first analyses one can do with their data, is to find the minimum,  maximum and the mean. The mean (or average) number of bikes per day rented in this case is the sum of all bikes rented per day divided by the total number of days:

$\bar{f} = \frac{\sum_{i=1}^{n} b_i}{n}$

where $\bar{f}$ is the mean, $b_i$ is the number of bikes rented on day $i$ and $n$ are the total number of days. We can compute these in python using the following code:

{% highlight python %}

from datetime import datetime
from collections import defaultdict
import csv
import numpy

data = defaultdict(int)
prevdate = ""
with open("bikesharing.csv", "r") as fin:
    csvreader = csv.reader(fin)
    next(csvreader)
    for row in csvreader:
        date_object = datetime.strptime(row[0].split(" ")[0], '%m/%d/%y')
        currdate = str(date_object.month) + "-" + str(date_object.day)
        # Computing the total number of bikes rented in a day
        data[currdate] += int(row[4])

totals = []
for key, value in data.iteritems():
    totals.append(value)

numpy.min(totals)
numpy.max(totals)
numpy.mean(totals)

output:
2752 #min
13446 #max
9146.82 #mean

{% endhighlight %}

In this case, the mean is 9146.82. It looks like there are several large values in the data, because the mean is closer to the max than to the min. Maybe the data will provive more insight if we compute the min, max and mean per day, grouped by another factor, like season or weather. Here is some code to compute mean per day grouped by the season:

{% highlight Python %}

from datetime import datetime
import csv
import numpy

data = {}

with open("bikesharing.csv", "r") as fin:
    csvreader = csv.reader(fin)
    next(csvreader)
    for row in csvreader:
        date_object = datetime.strptime(row[0].split(" ")[0], '%m/%d/%y')
        currdate = str(date_object.month) + "-" + str(date_object.day)
        if row[1] in data:
            if currdate in data[row[1]]:
                data[row[1]][currdate] += int(row[4])
            else:
                data[row[1]].update({ currdate : int(row[4])})
        else:
            data[row[1]] = { currdate : int(row[4]) }


for key, value in data.iteritems():
    print key
    totals = []
    for k, val in value.iteritems():
        totals.append(val)
    print numpy.min(totals)
    print numpy.max(totals)
    print numpy.mean(totals)


output:
1 #season
2752 #min
10580 #max
5482.42105263 #mean

2
6252
13088
10320.7368421

3
7818
13446
11239.6842105

4
5713
12982
9544.45614035

{% endhighlight %}

As you can see, the mean varies significantly with the season. It intuitively makes sense because we would expect more people to ride a bike in the summer as compared to the winter, which means a higher mean in the summer than the winter, this also means higher min and max values in the summer than the winter. This data also helps us intuitive guess that season 1, is most likely winter, and season 3 is most likely summer.

---

## VARIABILITY IN DATA

The next thing we would like to know, is the variability of the data provided. It is good to know if the data is skewed in a particular direction, or how varied it is. If the data is highly variable, it is hard to determine if the mean changes with different samples of data. Reducing variability is a common goal of designed experiments, and this can be done by finding subsets of data that have low variablity such that samples from each of the subsets produce similar mean value. We already did a little bit of that in the second example above.

There are 2 ways of measuring variability: variance and standard deviation.

## VARIANCE

Variance is defined as the average of the squared differences from the mean. In most experiments, we take a random sample from a population. In this case, we will compute the population variance, which uses all possible data provided. Population variance can be computed as:

$$ \sigma^2 = \frac{\sum_{i=1}^{n} (f_i - \bar{x})^2}{N} $$

If you needed to compute the sample variance, you can use the following formula:

$$variance = \frac{\sum_{i=1}^{n} (f_i - \bar{x})^2}{N-1}$$

where $x_i $ is each instance, $\bar{x}$ is the mean, and $N$ is the total number of features. Dividing by n−1 gives a better estimate of the population standard deviation for the larger parent population than dividing by n, which gives a result which is correct for the sample only. This is known as Bessel's correction.

In our case we will compute population variance using most of the same code as that above, except adding the following line to it:

{% highlight Python %}

print numpy.var(totals)

Output:
1 #season
2752 #min
10580 #max
5482.42105263 #mean
2812044.87535 #variance

2
6252
13088
11239.6842105
2953435.21145

3
7818
13446
11239.6842105
1368005.2687

4
5713
12982
9544.45614035
2552719.23053

{% endhighlight %}

## STANDARD DEVIATION

Variance by itself is not particularly insightful, as its units are feature squared and it is not possible to plot it on a graph and compare it with the min, max and mean values. The square root of variance is the standard deviation, and it is a much more insightful metric.

The population standard deviation, $\sigma$, is the square root of the variance, $\sigma^2$. In python you can compute variance by adding the following line to the above code:

{% highlight Python %}

print numpy.var(totals)

output:

1 #season
2752 #min
10580 #max
5482.42105263 #mean
1676.91528568 #standard deviation

2
6252
13088
10320.7368421
1718.55614149

3
7818
13446
11239.6842105
1169.6175737

4
5713
12982
9544.45614035
1597.72313951
{% endhighlight %}

A standard deviation close to 0 indicates that the data points tend to be very close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the data points are spread out over a wider range of values. In our case, the data is very spread out. Three standard deviations from the mean account for 99.7% of the sample population being studied, assuming the distribution is normal (bell-shaped).

---

## STANDARD ERROR OF THE MEAN (SEM)

In this post, we have computed the population mean, however, if one has to compute the sample mean, it is useful to know how accurate this value is in estimating the population mean. SEM is the error in estimating $\mu$.

$$SEM = \frac{\sigma}{\sqrt{N}}$$

however, as we often are unable to compute the population standard deviation, we will use teh sample standard deviation instead:

$$SEM = \frac{s}{\sqrt{N}}$$

The mean of any given sample is an estimate of the population mean number of features. Two aspects of the population and the sample could affect the variability of the mean number of features of those samples.

- If the population of number of features has very small standard deviation, then the samples from that population will have small sample standard deviation the sample means will be close to the population mean and we will have a small standard error of the mean
- If the population of number of features has a large standard deviation, then the samples from that population will have large sample standard deviation the sample means may be far from the population mean and we will have a large standard error of the mean

So large population variability causes a large standard error of the mean. The estimate of the population mean using 2 observations is less reliable than the estimate using 20 observations, and much less reliable than the estimate of the mean using 100 observations. As N gets bigger, we expect our error in estimating the population mean to get smaller.
