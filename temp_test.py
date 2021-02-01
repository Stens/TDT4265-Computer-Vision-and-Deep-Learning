import numpy as np


l = [1,2,3]
a = np.array([l,l,l])
print(a)
print("----------------------")
for i in range(a.shape[0]):
    for j in range(a[i].shape[0]):
        a[i][j] = a[i][j] + 1
print(a)        

# Add bias trick



b = np.array([[1]*3])

a = np.concatenate((a,b.T),axis=1)
print("---------------------")
print(a)

