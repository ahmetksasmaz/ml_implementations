import numpy as np

# (2,3)
a = np.array([
    [1,2,3],
    [4,5,6]
    ])

# (3,4)
b = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
])

# expecting (2,4)
print(np.matmul(a,b))