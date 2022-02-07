# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:56:11 2022

@author: Katherine Liu
"""

pip install wooldridge


import wooldridge as woo
import numpy as np
import scipy.stats as stats


audit = woo.dataWoo('audit')
y= audit['y']

avgy = np.mean(y)
n= len(y)
sdy = np.std(y, ddof =1)
se = sdy/np.sqrt(n)
c99 = stats.norm.ppf(0.995)


lowerCI99 = avgy -c99*se
print(lowerCI99)

upperCI99 = avgy +c99*se
print(upperCI99)
