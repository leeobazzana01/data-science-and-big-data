# Exploratory Data Analysis & Graphical Visualization Review

This report presents a theoretical and practical decomposition of the graphical methods implemented within the data science pipeline. The study is focused entirely on the **Employee Hourly Wages** dataset, utilizing visual statistical summaries to uncover data distributions, central tendencies, and categorical relationships.

## 1. Core Mathematical and Visual Frameworks

The pipeline leverages geometric and spatial representations to evaluate the statistical properties of the dataset.

### 1.1 Frequency Distributions (Histograms)

A histogram is a graphical representation of the distribution of numerical data. It acts as an estimator for the probability distribution of a continuous variable.

#### Mathematical Concept

The data range is divided into a set of contiguous, non-overlapping intervals called **bins**. The height of each bin is proportional to the absolute frequency ($n_i$) of observations falling within that interval.

If normalized, the area of the histogram approximates the **Probability Density Function (PDF)**.

---

### 1.2 Categorical Proportions (Pie Charts)

Pie charts map the relative frequency of discrete categorical variables onto a circular geometry.

#### Mathematical Concept

The area (and consequently, the central angle $\theta$) of each slice is strictly proportional to the relative frequency of the category.

$$
\theta_i = \left( \frac{n_i}{N} \right) \times 360^\circ
$$

---

### 1.3 Quartile Distribution & Dispersion (Boxplots)

Box-and-whisker plots are standardized ways of displaying the distribution of data based on a five-number summary:

- Minimum
- First Quartile ($Q_1$)
- Median ($Q_2$)
- Third Quartile ($Q_3$)
- Maximum

#### Interquartile Range (IQR)

Measures the statistical dispersion of the middle 50% of the data. It is a robust measure of variability.

$$
IQR = Q_3 - Q_1
$$

#### Outlier Detection

The "whiskers" extend to the furthest data points that are not considered outliers. A common mathematical boundary for detecting potential outliers uses the IQR:

$$
\text{Lower Bound} = Q_1 - 1.5 \times IQR
$$

$$
\text{Upper Bound} = Q_3 + 1.5 \times IQR
$$

---

## 2. Dataset Analysis Breakdowns

### 2.1 Continuous Feature Distributions

Using histograms, the continuous variables `education_yrs`, `experience_yrs`, and `age` were plotted.

#### Visual Insight

The histograms for `age` and `experience_yrs` visually confirm the positive skewness (right-skewed distribution) mathematically calculated in previous parametric analyses.

A heavy concentration of observations falls within the lower bounds of the distributions (e.g., younger workers and fewer years of experience), while a long tail extends toward higher values.

This pattern is characteristic of positively skewed distributions:

$$
\bar{x} > \tilde{x} > \text{Mode}
$$

---

### 2.2 Categorical Segmentations

Using a grid of subplots, binary categorical variables (dummy variables) were visualized through pie charts:

- `union`
- `female`
- `marr`
- `south`
- `manufacturing`
- `construction`

#### Visual Insight

These charts provide immediate comprehension of class proportions and potential imbalances within the dataset.

Examples include:

- Male vs. female workforce composition.
- Unionized vs. non-unionized workers.
- Married vs. unmarried individuals.
- Industrial sector participation.

The visual proportions establish baseline probabilities that may later influence predictive machine learning models, especially in classification and regression tasks.

---

### 2.3 Comparative Bivariate Analysis (Segmented Boxplots)

The most informative visualizations in the notebook arise from comparing continuous variables against categorical groupings using segmented boxplots.

---

#### Age Segmented by Gender (`female`)

The boxplots display the age distribution separately for:

- Men (`female = 0`)
- Women (`female = 1`)

##### Statistical Findings

Women exhibit slightly greater age variability than men.

The variance calculations confirm this observation:

- Female variance: **143.99**
- Male variance: **130.90**

Additionally:

- Men's first quartile ($Q_1$): **27 years**
- Women's first quartile ($Q_1$): **28 years**

This indicates a slightly older lower-bound age profile among women in the sample.

---

#### Hourly Wage Segmented by Gender (`female`)

##### Visual Insight

The boxplot reveals a noticeable difference in wage distributions between genders.

Key observations include:

- The median hourly wage ($Q_2$) for women is lower than that of men.
- The upper wage distribution for men extends significantly higher.
- The third quartile ($Q_3$) is substantially larger for men.

These graphical patterns suggest the presence of a wage gap within the dataset.

From a machine learning perspective, the variable `female` is therefore expected to contribute meaningful predictive information when modeling `wage_per_hour`.

---

#### Hourly Wage Segmented by Marital Status (`marr`)

The boxplots compare wage distributions between:

- Unmarried workers (`marr = 0`)
- Married workers (`marr = 1`)

##### Visual Insight

Married individuals exhibit a higher median wage than unmarried individuals.

Additional evidence includes:

- A larger third quartile ($Q_3$) among married workers.
- Greater concentration of higher-income observations within the married category.

This suggests that higher earnings in the dataset are positively associated with marital status, indicating that `marr` may also be an important explanatory variable in predictive wage modeling.

---

## Conclusion

The graphical exploratory analysis complements the numerical statistical analysis by providing intuitive visual evidence of distributional characteristics, variability, and group-level differences.

The results reveal:

- Positive skewness in age and experience distributions.
- Meaningful categorical imbalances across workforce segments.
- Higher age variability among women.
- Observable wage disparities by gender.
- Higher wage distributions among married workers.

Together, these findings establish a strong foundation for subsequent feature engineering, hypothesis testing, and machine learning regression models targeting hourly wage prediction.
