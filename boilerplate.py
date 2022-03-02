# BT2101 code
import wooldridge as woo
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# How to read csv (Convert to dataframe)
company = pd.read_csv("1999_company.csv")
# Show first five rows
company.head()

# How to read csv (Convert to tuple of tuple)


def read_csv(csvfilename):
    rows = ()
    with open(csvfilename, 'r', newline='') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows += (tuple(row), )
    return rows


# If need to use wooldridge dataset
audit = woo.dataWoo("audit")


# Calculate mean (Data is list)
np.mean(data)

# Calculate variance
np.var(data, ddof=1)

# Calculate standard deviation
np.std(data, ddof=1)


# t-test
# Find rejection region from critical value
# To find Pr(X > 1,5) where df = 10
1 - stats.t.cdf(1.5, 10)
# To find Pr(X < -1.5) where df = 10
stats.t.cdf(-1.5, 10)

# Find critical value from rejection region
# Find x such that Pr(X <= x) = 0.05
stats.t.ppf(0.05, 10)
# Find x such that Pr(X >= x) = 0.05
stats.t.ppf(0.95, 10)


# Normal distribution
# Find rejection region from critical value
# To find Pr(X <= 45), where mean = 50 and sd = 10
stats.norm.cdf(45, 50, 10)
# To find Pr(X > 45), where mean = 50 and sd = 10
1 - stats.norm.cdf(45, 50, 10)
# Find critical value from rejection region
# Find x such that Pr(X <= x) = 0.05, where mean = 50 and sd = 10
stats.norm.ppf(0.05, 50, 10)
# Find x such that Pr(X >= x) = 0.05, where mean = 50 and sd = 10
stats.norm.ppf(0.95, 50, 10)

# Confidence interval
lower = mean - (critical value * sd / np.sqrt(n))
upper = mean + (critical value * sd / np.sqrt(n))


# Convert to dataframe
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df = pd.DataFrame({'y-axis': y, 'x-axis': x})


# Linear regression model (Simple)
lm = smf.ols("y~x", df).fit()
# Gives you the coefficient of the intercept and the independent variable
lm.params()
# Gives you all the values you need to do stuff
lm.summary()

# Linear regression model (Multiple)
lm = smf.ols("y~x1+x2", data=df).fit()
lm.params()
lm.summary()


# Plot graph (One)
plt.plot('y', 'x', data=df, color='grey', marker='o', linestyle='')
plt.plot(df["x"], lm.fittedvalues, color='black', linestyle='-')
plt.ylabel('y-axis')
plt.xlabel('x-axis')

# Plot graph (Multiple)
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.scatter(univ['averhour'], univ['mbagpa'])
plt.xlabel('averhour')
plt.ylabel('mbagpa')
plt.subplot(1, 3, 2)
plt.scatter(univ['salary'], univ['mbagpa'])
plt.xlabel('salary')
plt.ylabel('mbagpa')
plt.subplot(1, 3, 3)
plt.scatter(univ['numofact'], univ['mbagpa'])
plt.xlabel('numofact')
plt.ylabel('mbagpa')
plt.show()
