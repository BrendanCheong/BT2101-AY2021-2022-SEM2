# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:28:56 2022

@author: Katherine Liu
"""



import numpy as np
import scipy as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm


data = [2.3,2.4,3.1,2.2,1.0,2.3,2.1,1.1,1.2,0.9,1.5,1.1]

print("mean:", np.mean(data))
print("var:", np.var(data,ddof=1))
print("std:", np.std(data,ddof=1))


n = len(data)
m = np.mean(data)
sd = np.std(data, ddof = 1)
cri = sp.stats.norm.ppf(loc = 0, scale = 1, q = 0.975)
lower=m-cri*sd/np.sqrt(n)
upper=m+cri*sd/np.sqrt(n)
print(lower)
print(upper)