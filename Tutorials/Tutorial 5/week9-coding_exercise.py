#BT2101 Class Material

# 1. Simulation model
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

np.random.seed(1234567)

y = stats.binom.rvs(1, 0.5, size=100)
x = stats.norm.rvs(0, 1, size=100) +2*y
sim_data = pd.DataFrame({'y': y, 'x': x})

reg_lin = smf.ols(formula='y~x', data=sim_data)
results_lin = reg_lin.fit()
reg_logit = smf.logit(formula='y~x', data=sim_data)
results_logit = reg_logit.fit(disp=0)
reg_probit = smf.probit(formula='y~x', data=sim_data)
results_probit = reg_probit.fit(disp=0)

X_new = pd.DataFrame ({'x': np.linspace(min(x), max(x), 50)})
predictions_lin = results_lin.predict(X_new)
predictions_logit = results_logit.predict(X_new)
predictions_probit = results_probit.predict(X_new)

plt.plot (x, y, color='grey', marker='o', linestyle='')
plt.plot (X_new['x'], predictions_lin, color='black', linestyle='-.', label='linear')
plt.plot (X_new['x'], predictions_logit, color='black', linestyle='-', linewidth=0.5, label='logit')
plt.plot (X_new['x'], predictions_probit, color='black', linestyle='--', label='probit')
plt.ylabel('y')
plt.xlabel('x')
plt.xlabel('x')
plt.legend()

# 2. Linear model
import wooldridge as woo
import pandas as pd
import statsmodels. formula.api as smf

mroz = woo.dataWoo('mroz')

reg_lin = smf.ols(formula = 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6', data=mroz)
results_lin = reg_lin.fit(cov_type='HC3')

table = pd.DataFrame ({'b': round(results_lin.params, 3),'se': round(results_lin.bse, 3),'t': round(results_lin.tvalues, 3), 
                     'pval': round(results_lin.pvalues, 3)})
print (f'table: \n{table}\n')

X_new = pd.DataFrame (
    {'nwifeinc': [100, 0], 'educ': [5, 17], 'exper': [0,30] 
     ,'age': [20,52],'kidslt6': [2,0], 'kidsge6': [0,0]})
predictions = results_lin.predict(X_new)
print (f'predictions: \n{predictions}\n')


# 3. Probit model

import wooldridge as woo
import pandas as pd
import statsmodels. formula.api as smf

mroz = woo.dataWoo('mroz')

reg_probit = smf.probit(formula = 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6', data=mroz)
results_probit = reg_probit.fit(disp=0)
print (f'results_probit.summary(): \n{results_probit.summary()}\n')

print (f'results_probit.llf: {results_probit.llf}\n')
print (f'results_probit.prsquared: {results_probit.prsquared}\n')


# 4. Logit model

import wooldridge as woo
import pandas as pd
import statsmodels. formula.api as smf

mroz = woo.dataWoo('mroz')

reg_logit = smf.logit(formula = 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6', data=mroz)
results_logit = reg_logit.fit(disp=0)
print (f'results_logit.summary(): \n{results_logit.summary()}\n')

print (f'results_logit.llf: {results_logit.llf}\n')
print (f'results_logit.prsquared: {results_logit.prsquared}\n')

