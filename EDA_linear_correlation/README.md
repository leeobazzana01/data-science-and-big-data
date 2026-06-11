# Exploratory Data Analysis & Correlation Metrics Review

This report presents a theoretical and practical decomposition of the bivariate and multivariate statistical operations implemented within the data science pipeline. The study focuses on correlation analysis and linear estimation across three distinct datasets: **Advertising ROI**, **Restaurant Tips**, and **Socioeconomic Indicators (Atlas Brasil 2014)**.

## 1. Core Mathematical and Data Science Frameworks

Moving beyond one-dimensional descriptive statistics, this phase explores how two or more variables interact, covary, and influence one another.

### 1.1 Bivariate Relationships and Scatterplots

A scatterplot is the primary geometric tool for visualizing the relationship between two continuous variables.

By plotting one feature on the Cartesian **x-axis** (independent variable) and another on the **y-axis** (dependent variable or target), data scientists can visually detect:

- Patterns
- Clusters
- Trends
- Covariance structures
- Potential outliers

Scatterplots are often the first step in determining whether a predictive relationship may exist between variables.

---

### 1.2 Pearson Correlation Coefficient ($r$)

The Pearson correlation coefficient measures the **linear strength and direction** of the association between two continuous variables.

#### Mathematical Concept

It is calculated as the covariance of the two variables divided by the product of their standard deviations.

$$
r =
\frac{
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
}{
\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}
}
$$

#### Interpretation Scale

The value of $r$ always falls within the interval:

$$
-1 \le r \le 1
$$

| Correlation Value | Interpretation |
|------------------|----------------|
| $r \approx 1$ | Strong positive linear correlation |
| $r \approx 0$ | No linear correlation |
| $r \approx -1$ | Strong negative linear correlation |

#### Domain Standard

In practical data science applications:

$$
0.3 \le |r| \le 0.7
$$

typically indicates a **moderate relationship**, while values approaching 1 indicate increasingly strong linear associations.

---

### 1.3 Simple Linear Regression (Polynomial Fit of Degree 1)

When correlation is established, linear regression enables the estimation or prediction of a dependent variable ($y$) from an independent variable ($x$).

#### Mathematical Concept

The **Ordinary Least Squares (OLS)** method determines the line of best fit by minimizing the sum of squared residuals.

Residuals represent the vertical distances between observed data points and the estimated regression line.

The resulting linear model is:

$$
y = \beta_1 x + \beta_0
$$

where:

- $\beta_1$ = slope (angular coefficient)
- $\beta_0$ = intercept

The function:

```python
numpy.polyfit(x, y, 1)
```

extracts these coefficients algorithmically.

The generated coefficients can then be encapsulated into a predictive function using:

```python
numpy.poly1d()
```

creating a reusable regression estimator.

---

## 2. Dataset Analysis Breakdowns

### 2.1 Dataset 1: Advertising ROI

**Objective:** Determine which advertising channel (`TV`, `Radio`, or `Newspaper`) has the strongest influence on the target variable `sales`.

#### Analytical Insights

Scatterplots were generated for each advertising medium against sales, with regression lines superimposed to evaluate predictive strength.

##### Strongest Influence: TV Advertising

The `TV` versus `Sales` plot exhibits:

- Tight clustering around the regression line.
- Strong positive directionality.
- A relatively steep slope.

These characteristics indicate a strong positive correlation between television advertising expenditure and sales performance.

As a result:

$$
\text{TV} \rightarrow \text{Best Predictor of Sales}
$$

##### Weakest Influence: Newspaper Advertising

The `Newspaper` versus `Sales` plot displays:

- Significant dispersion of observations.
- Large residual distances.
- Weak alignment with the regression line.

This pattern suggests a weak or negligible linear relationship between newspaper advertising investments and sales outcomes.

Thus:

$$
\text{Newspaper} \rightarrow \text{Lowest Predictive Power}
$$

---

### 2.2 Dataset 2: Restaurant Tips Estimation

**Objective:** Analyze the relationship between restaurant bill amounts (`total_bill`) and tip values (`tip`) to build a predictive estimator.

#### Statistical Outputs

##### Pearson Correlation Evaluation

The calculated Pearson coefficient was:

$$
r = 0.6757
$$

#### Interpretation

This value indicates a **positive moderate-to-strong correlation**.

As the total bill increases, tip values tend to increase proportionally.

Mathematically:

$$
r > 0
\Rightarrow
\text{Positive Association}
$$

and

$$
|r| \approx 0.68
\Rightarrow
\text{Moderately Strong Relationship}
$$

---

#### Algorithmic Estimation

The notebook employed:

```python
numpy.polyfit(total_bill, tip, 1)
```

to estimate:

- $\beta_1$ (slope)
- $\beta_0$ (intercept)

These coefficients were then encapsulated within:

```python
numpy.poly1d()
```

effectively transforming the regression equation into a callable predictive model.

The resulting estimator behaves as:

$$
\widehat{\text{tip}}
=
\beta_1(\text{total\_bill})
+
\beta_0
$$

where:

- Input: `total_bill`
- Output: Predicted `tip`

This constitutes a simple machine learning regression model.

---

### 2.3 Dataset 3: Atlas Brasil 2014 (Multivariate Overview)

**Objective:** Conduct a macro-level correlation analysis across multiple socioeconomic indicators:

- `MORT1`
- `ANOSEST`
- `T_ANALF25M`
- `RDPC`
- `IDHM`

---

#### Multivariate Visualization Strategy

Rather than manually creating pairwise plots, the project utilized:

```python
seaborn.pairplot(kind='reg')
```

#### Data Science Insight

This function automatically generates a **Scatterplot Matrix**, displaying every numerical variable against every other numerical variable in the dataset.

For each pair of features, the visualization includes:

- Scatterplot observations.
- Linear regression trend line.
- Distributional summaries along the diagonal.

Conceptually, for a dataset containing $n$ variables, the matrix explores:

$$
n \times n
$$

relationships simultaneously.

---

#### Practical Importance

Scatterplot matrices are fundamental in professional exploratory data analysis because they allow analysts to rapidly identify:

##### Feature Correlations

Variables that move together strongly.

$$
|r| \rightarrow 1
$$

##### Multicollinearity

Situations where predictor variables contain redundant information, potentially harming regression model stability.

##### Candidate Predictors

Variables exhibiting strong relationships with the target feature.

##### Nonlinear Structures

Patterns that may not be captured by linear models and may require more advanced machine learning approaches.

---

## Conclusion

This phase of exploratory analysis extends beyond descriptive statistics into the domain of statistical relationships and predictive modeling.

The key findings include:

- **TV advertising** is the strongest predictor of sales performance.
- **Newspaper advertising** exhibits minimal predictive influence.
- **Restaurant bill totals and tips** possess a moderate-to-strong positive correlation ($r = 0.6757$).
- Linear regression successfully transforms observed relationships into predictive mathematical models.
- Scatterplot matrices provide an efficient framework for identifying correlation structures, multicollinearity, and high-value predictors within multivariate datasets.

Together, these methods form the statistical foundation of supervised machine learning workflows, where understanding variable interactions is essential for building accurate predictive models.
