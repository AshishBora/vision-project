
# coding: utf-8

# In[1]:

import numpy as np


# In[3]:

def getTopK(a, k):
    return sorted(range(len(a)), key = lambda i: a[i])[-k:]
# A = [1, 2, 5, 3, 4, 2, 1, 3, 4]
# indices = getTopK(A, 3)
# print(indices)


# In[5]:

def getNonZero(a):
    return [i for i in range(len(a)) if a[i]>0]
# A = [1, 0, 5, -3, 4, -2, -1, 0, 4]
# indices = getNonZero(A)
# print(indices)


# In[6]:

def getIntersection(P, T):
    return list(set(P) & set(T))
# T = [6, 2, 5, 1]
# P = [1, 5, 7, 8]
# intersection = getIntersection(T, P)
# print(intersection)


# In[10]:

def getPrecision(prob, target):
    precision = 0
    num_im = target.shape[0]
    print('Total number of test images =', num_im)

    for i in range(num_im):
        trgt = target[i, :]
        T = getNonZero(trgt)
        k = len(T)
        prb = prob[i, :]
        P = getTopK(prb, k)
        intersection = getIntersection(T, P)
        prec = len(intersection) * 1.0 / k
        precision = precision + prec

    precision = precision * 1.0 / num_im
    return precision


# In[11]:

target = np.load('./npy/probs_test_gt.npy')
prob = np.load('./npy/probs_test.npy')

print(target.shape)
print(prob.shape)


# In[12]:

precision = getPrecision(prob, target)
print('Average precision = ', precision)


# In[ ]:



