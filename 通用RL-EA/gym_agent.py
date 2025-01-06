import matplotlib.pyplot as plt 
import numpy as np
n = 10000000 
l1 = [] 
l2 = []
s = 0
for i in range(1, n+1):
    s = s + 1/i
    l1.append(s)
    l2.append(np.log(i))

plt.plot(l1,label='l1')
plt.plot(l2,label='l2')
plt.legend()
plt.show()
