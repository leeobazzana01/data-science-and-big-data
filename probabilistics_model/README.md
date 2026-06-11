# Probabilistic Models & Statistical Distributions Review

This report presents a theoretical and practical decomposition of the probabilistic frameworks implemented in the data science pipeline. The study is divided into two primary categories: **Discrete Probability Distributions** (Binomial and Poisson) and **Continuous Probability Distributions** (Normal Distribution and Z-Scores).

---

## 1. Core Mathematical Frameworks: Discrete Distributions

Discrete probability models are used when the random variable can only take on specific, distinct values (e.g., counts, integers).

### 1.1 The Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent binary trials, where each trial results in either:

- Success
- Failure

#### Mathematical Concept

The probability of obtaining exactly $k$ successes in $n$ independent trials, given a constant success probability $p$ and failure probability $q$ (where $q = 1-p$), is:

$$
P(X=k)
=
\binom{n}{k}
p^k
q^{n-k}
$$

where the binomial coefficient is defined as:

$$
\binom{n}{k}
=
\frac{n!}{k!(n-k)!}
$$

#### Assumptions

A Binomial model requires:

1. A fixed number of trials ($n$).
2. Independent trials.
3. Only two possible outcomes per trial.
4. Constant probability of success ($p$).

---

### 1.2 The Poisson Distribution

The Poisson distribution models the probability of a given number of events occurring within a fixed interval of time or space.

Typical examples include:

- Manufacturing defects.
- Incoming customer requests.
- Website visits.
- Equipment failures.

#### Mathematical Concept

Assuming a known average event rate ($\lambda$), the probability of observing exactly $k$ events is:

$$
P(X=k)
=
\frac{
e^{-\lambda}\lambda^k
}{k!}
$$

where:

- $e \approx 2.71828$
- $\lambda$ = expected number of occurrences
- $k$ = observed number of events

#### Key Property

The Poisson distribution is completely determined by a single parameter:

$$
E(X) = \lambda
$$

and

$$
Var(X) = \lambda
$$

meaning that its mean and variance are equal.

---

## 2. Core Mathematical Frameworks: Continuous Distributions

Continuous probability models are used when the random variable can assume any value within an interval.

Examples include:

- Time
- Weight
- Height
- Income
- Age

---

### 2.1 The Normal (Gaussian) Distribution

The Normal distribution is one of the most important probability models in statistics and machine learning.

It is characterized by:

- Mean ($\mu$)
- Standard deviation ($\sigma$)

and forms the familiar bell-shaped curve.

#### Probability Density Function

$$
f(x)
=
\frac{1}
{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

#### Key Characteristics

- Perfectly symmetric around the mean.
- Mean, median, and mode are identical.
- Fully determined by $\mu$ and $\sigma$.

---

### The Empirical Rule (68–95–99.7 Rule)

A fundamental property of normally distributed data states:

| Range | Percentage of Data |
|---------|------------------:|
| $\mu \pm 1\sigma$ | 68% |
| $\mu \pm 2\sigma$ | 95% |
| $\mu \pm 3\sigma$ | 99.7% |

This rule provides a quick approximation of probability intervals without requiring numerical integration.

---

### 2.2 Standardization and Z-Scores

To compare observations from different normal distributions or calculate exact probabilities, values are standardized.

#### Mathematical Concept

The standardized score (Z-score) measures how many standard deviations a value lies from the mean.

$$
Z
=
\frac{X-\mu}{\sigma}
$$

where:

- $X$ = observed value
- $\mu$ = population mean
- $\sigma$ = population standard deviation

#### Interpretation

| Z-Score | Meaning |
|----------|---------|
| $Z = 0$ | Exactly at the mean |
| $Z > 0$ | Above the mean |
| $Z < 0$ | Below the mean |

After standardization, probabilities can be obtained using the **Standard Normal Distribution**:

$$
N(0,1)
$$

and its cumulative distribution function (CDF).

---

# 3. Practical Problem Breakdowns

## 3.1 Genetics Problem (Binomial Distribution)

### Objective

Calculate the probability that parents have exactly **3 blonde-haired children** out of **6 children**, given that each child independently has a 25% probability of being blonde.

### Parameters

$$
n=6
$$

$$
k=3
$$

$$
p=0.25
$$

$$
q=0.75
$$

---

### Mathematical Solution

The combinatorial component is:

$$
\binom{6}{3}
=
20
$$

Applying the Binomial formula:

$$
P(X=3)
=
\binom{6}{3}
(0.25)^3
(0.75)^3
$$

Result:

$$
P(X=3)
\approx
0.1318
$$

or:

$$
13.18\%
$$

---

### Python Implementation

The notebook utilized:

```python
scipy.stats.binom.pmf()
```

to directly compute the probability mass function (PMF), avoiding manual factorial calculations and enabling graphical visualization of the distribution.

---

## 3.2 Manufacturing Defect Problem (Poisson Distribution)

### Objective

Calculate the probability of observing exactly **4 defective screws per hour**, given an average failure rate of **8 defects per hour**.

### Parameters

$$
\lambda = 8
$$

$$
k = 4
$$

---

### Mathematical Solution

Applying the Poisson formula:

$$
P(X=4)
=
\frac{
e^{-8}8^4
}{4!}
$$

Result:

$$
P(X=4)
\approx
0.05725
$$

or:

$$
5.73\%
$$

---

### Data Science Insight

The PMF visualization reveals that:

- The highest probability mass naturally concentrates near the mean.
- Since $\lambda = 8$, the most likely number of defects occurs around 8 events.
- Observing only 4 defects lies on the lower-left tail of the distribution.

This illustrates how Poisson probabilities decrease as observations move away from the expected rate.

---

## 3.3 Retirement Demographics (Empirical Rule)

### Objective

Determine age ranges containing the majority of a retiring population.

### Parameters

$$
\mu = 64
\text{ years}
$$

$$
\sigma = 3
\text{ years}
$$

---

### 68% Confidence Bracket

Using:

$$
\mu \pm 1\sigma
$$

we obtain:

$$
64 \pm 3
$$

Result:

$$
61 \le X \le 67
$$

Approximately:

$$
68\%
$$

of the population falls within this interval.

---

### 95% Confidence Bracket

Using:

$$
\mu \pm 2\sigma
$$

we obtain:

$$
64 \pm 6
$$

Result:

$$
58 \le X \le 70
$$

Approximately:

$$
95\%
$$

of the population falls within this interval.

---

### Specific Range: 55 to 67 Years

This interval spans:

$$
-3\sigma
\rightarrow
+1\sigma
$$

within the standard normal framework.

Using the empirical distribution segments:

$$
P(55 \le X \le 67)
\approx
83.85\%
$$

Thus, approximately 83.85% of retirees fall within this age range.

---

## 3.4 Computer Usage Constraints (Z-Scores & CDF)

### Objective

Compute exact probabilities for daily computer usage using the cumulative distribution function (CDF).

### Parameters

$$
\mu = 5
\text{ hours}
$$

$$
\sigma = 1
\text{ hour}
$$

---

### Case 1: Less Than 2.5 Hours

#### Standardization

$$
Z
=
\frac{2.5-5}{1}
=
-2.5
$$

Using the standard normal table:

$$
P(X<2.5)
=
P(Z<-2.5)
\approx
0.0062
$$

Result:

$$
0.62\%
$$

---

### Case 2: Between 2.5 and 7.5 Hours

The interval corresponds to:

$$
-2.5 \le Z \le 2.5
$$

Using cumulative probabilities:

$$
P(Z \le 2.5)
=
0.9938
$$

$$
P(Z \le -2.5)
=
0.0062
$$

Therefore:

$$
P(2.5<X<7.5)
=
0.9938 - 0.0062
$$

Result:

$$
0.9876
$$

or:

$$
98.76\%
$$

---

### Case 3: More Than 7.5 Hours

#### Standardization

$$
Z
=
\frac{7.5-5}{1}
=
2.5
$$

Because the normal distribution is symmetric:

$$
P(Z>2.5)
=
P(Z<-2.5)
$$

Thus:

$$
P(X>7.5)
=
0.0062
$$

Result:

$$
0.62\%
$$

---

# Conclusion

This analysis explored the theoretical foundations and practical applications of both discrete and continuous probability distributions.

The primary findings include:

- The **Binomial Distribution** models fixed numbers of independent success/failure trials.
- The **Poisson Distribution** models event counts occurring at a known average rate.
- The **Normal Distribution** provides a powerful framework for modeling continuous variables.
- **Z-score standardization** enables exact probability calculations using the standard normal distribution.
- Python libraries such as `scipy.stats` automate these calculations while preserving mathematical rigor.

Together, these probabilistic tools form a foundational component of statistical inference, predictive modeling, risk analysis, and machine learning workflows.
