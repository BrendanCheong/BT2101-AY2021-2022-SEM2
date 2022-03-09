#!/usr/bin/env python
# coding: utf-8

# Week 9 Coding example I

import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def scale(x):
    x_mean = np.mean(x)
    x_var = np.var(x, ddof=1)
    x_scaled = (x-x_mean)/ np.sqrt(x_var)
    return x_scaled

hprice2 = woo.dataWoo('hprice2')
hprice2['price_sc'] = scale(hprice2['price'])
hprice2['nox_sc'] = scale(hprice2['nox'])
hprice2['crime_sc'] = scale(hprice2['crime'])
hprice2['rooms_sc'] = scale(hprice2['rooms'])
hprice2['dist_sc'] = scale(hprice2['dist'])
hprice2['stratio_sc'] = scale(hprice2['stratio'])

reg = smf.ols(formula = 'price_sc ~ 0 + nox_sc + crime_sc + rooms_sc + dist_sc + stratio_sc', data=hprice2)

results = reg.fit()

table = pd.DataFrame({'b': round(results.params, 3),'se': round(results.bse, 3),'t': round(results.tvalues, 3),'b': round(results.params, 3), 
                     'pval': round(results.pvalues, 3)})

print (f'table: \n{table}\n')


# Week 9 Coding example II

import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

hprice2 = woo.dataWoo('hprice2')

reg2 = smf.ols(formula = 'np.log(price) ~ np.log(nox) + np.log(dist) + rooms +I(rooms**2) + stratio', data=hprice2)

results2 = reg2.fit()

table2 = pd.DataFrame({'b': round(results2.params, 3),'se': round(results2.bse, 3),'t': round(results2.tvalues, 3),'b': round(results2.params, 3),
                     'pval': round(results2.pvalues, 3)})

print (f'table2: \n{table2}\n')


# Week 9 Coding example III

import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

attend = woo.dataWoo('attend')
n = attend.shape[0]

reg3 = smf.ols(formula = 'stndfnl ~ atndrte*priGPA + ACT + I(priGPA**2) + I(ACT**2)', data=attend)
results3 = reg3.fit()

table3 = pd.DataFrame({'b': round(results3.params, 3),'se': round(results3.bse, 3),'t': round(results3.tvalues, 3),'b': round(results3.params, 3),
                     'pval': round(results3.pvalues, 3)})

print (f'table3: \n{table3}\n')

