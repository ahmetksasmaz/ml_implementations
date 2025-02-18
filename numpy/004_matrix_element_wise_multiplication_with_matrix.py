import numpy as np

# (2,3)
a = np.array([
    [1,2,3],
    [4,5,6]
    ])

# (2,3)
b = np.array([
    [12,22,32],
    [41,51,61]
    ])

# expecting (2,3)
print(a*b)