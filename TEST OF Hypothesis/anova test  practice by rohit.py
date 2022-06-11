
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count  = np.array([9,8])
region = np.array([395,395])
stat,pval = proportions_ztest(count,region)
print('{0:0.3f}'.format(pval))

alpha = 0.05
#toh
#H0 = p1 = p2
#H1 = p2!= p2

if pval < alpha:
    print("H0 is rejected and H1 is accepted")
else:
    print("H1 is rejected amd H0 is accepted")
    

