# Exploratory Data Analysis & Descriptive Statistics Review

This report presents a theoretical and practical decomposition of the statistical operations implemented within the data science pipeline. The study is divided into two parts based on the ingested datasets: **Employee Hourly Wages** and **Socioeconomic Indicators (Atlas Brasil 2014)**.

## 1. Core Mathematical and Data Science Frameworks

### 1.1 Measures of Central Tendency

Central tendency metrics capture the "center" or typical value of a probability distribution or dataset.

#### Arithmetic Mean ($\mu$ or $\bar{x}$)
Computes the central location by summing all operational instances and dividing by the total population count ($N$ or $n$). It acts as the center of gravity for the data but is highly sensitive to extreme outliers.

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

#### Median ($\tilde{x}$)
The spatial middle value of an ordered dataset. If $n$ is odd, it is the exact middle element; if $n$ is even, it is the average of the two middle elements. It acts as a robust metric against skewed distributions.

$$
\tilde{x} =
\begin{cases}
x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\
\frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2} & \text{if } n \text{ is even}
\end{cases}
$$

#### Mode
The value or values that appear with the highest frequency within the dataset. A dataset can be unimodal, bimodal, multimodal, or amodal.

### 1.2 Measures of Dispersion and Relative Variability

Dispersion models quantify the spread or variability of data points around their central location.

#### Standard Deviation ($\sigma$ or $s$)
Measures the average absolute distance of data points from the sample mean. Expressed in the exact same units as the underlying feature.

$$
s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

#### Coefficient of Variation ($CV$)
A dimensionless measure of relative variability defined as the ratio of the standard deviation to the mean, scaled as a percentage.

**Data Science Insight:** Absolute standard deviation cannot be used to compare dispersion between two features if they have different measurement scales or significantly different means. The $CV$ normalizes the variance, allowing direct, scale-invariant comparisons.

$$
CV = \left( \frac{s}{\bar{x}} \right) \times 100\%
$$

### 1.3 Position Metrics: Quantiles and Percentiles

Quantiles split a sorted, continuous probability distribution or sample space into equal-sized partitions. The $p$-th percentile represents the value below which $p\%$ of the observations fall.

---

## 2. Dataset Analysis Breakdowns

### 2.1 Dataset 1: Hourly Wages Analysis

**Objective:** Analyze worker characteristics to prepare a structural foundation for machine learning regression tasks targeting `wage_per_hour`.

**Instance Count:** 534 instances with 10 explicit columns.

#### Feature Matrix Breakdown

The variables calculated in the script yield critical domain knowledge about the distribution of features:

| Feature Name | Mean ($\bar{x}$) | Median ($\tilde{x}$) | Mode |
|---|---:|---:|---:|
| Education (`education_yrs`) | 13.02 years | 12.00 years | 12.00 years |
| Experience (`experience_yrs`) | 17.82 years | 15.00 years | 14.00 years |
| Age (`age`) | 36.83 years | 35.00 years | 32.00 years |

#### Skewness Interpretation

For all three variables, $\bar{x} > \tilde{x} > \text{Mode}$. This systematically validates that `education_yrs`, `experience_yrs`, and `age` follow a **right-skewed (positively skewed)** distribution.

#### Positional Percentile Engineering

- **70th Percentile of Age:** Evaluated at **42.0 years**, which mathematically guarantees that 70% of the monitored workforce is $\leq 42.0$ years old.
- **10th Percentile of Experience:** Evaluated at **3.0 years**, meaning 10% of workers possess 3 years or less of domain exposure.

#### Relative Dispersion Selection

To determine which feature varies the most, absolute standard deviations ($s$) were mapped against coefficients of variation ($CV$):

- `education_yrs`: $s = 2.61 \implies CV = 20.09\%$
- `age`: $s = 11.72 \implies CV = 31.84\%$
- `experience_yrs`: $s = 12.38 \implies CV = 69.46\%$

$$
\text{Maximal Variability Location} = \operatorname{arg\,max}(CV) \implies \texttt{experience\_yrs}
$$

---

### 2.2 Dataset 2: Atlas Brasil 2014 Analysis

**Objective:** Statistically map demographic, educational, and economic conditions across Brazilian states.

#### Summary of Key Statistical Outputs

- **Mean Human Development Index ($\overline{\text{IDHM}}$):** Evaluated at **0.7376**, offering a clear macro-baseline of the country's socio-economic status for 2014.
- **Median Child Mortality (MORT1):** Evaluated at **16.86**. This shows that 50% of the states experience a child mortality rate below 16.86 per 1,000 live births.
- **Upper Quartile of Per Capita Income (RDPC):** The 75th percentile equals **R$ 885.16**. Because the prompt asked for the value where 25% of states present a superior value, we compute the right-tail inverse boundary $(1 - 0.25 = 0.75)$. Thus, exactly 25% of Brazilian states achieve an average per capita income higher than R$ 885.16.

#### Relative Dispersion Comparison

The script evaluated variance differences between `T_ANALF25M` (Adult Literacy Rate $\geq 25$ years old) and `RDPC` (Per Capita Income):

- `RDPC`: $CV = 37.28\%$
- `T_ANALF25M`: $CV = 59.00\%$

$$
\text{Maximal Variability Location} = \operatorname{arg\,max}(CV) \implies \texttt{T\_ANALF25M}
$$

**Conclusion:** Although the absolute standard deviation of Per Capita Income (`RDPC`) is numerically higher due to currency values, the normalized relative dispersion proof shows that Adult Illiteracy (`T_ANALF25M`) has significantly higher inequality/variance across Brazilian states.
