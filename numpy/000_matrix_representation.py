import numpy as np

a = np.array(
    [
        [[1,2,3],[4,5,6]],
        [[7,8,9],[10,11,12]],
        [[13,14,15],[16,17,18]],
        [[19,20,21],[22,23,24]]
    ])

print(a)
print(a.dtype)
print(a.shape) # len(list), len(list[0]), len(list[0][0]) ...