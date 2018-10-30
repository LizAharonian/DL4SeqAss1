import numpy as np
W = np.zeros((3, 4))
b = np.zeros(4)

liz = np.shape(W)[1]
print "liz"

liz = [11,1,22,2,33,3,44,4]


la = liz[0:-2:2]
le = liz[1:-1:2]

for w,b in zip(liz[0:-2:2], liz[1:-1:2]):
    print w,b

for i, (W_i, b_i) in enumerate(zip(liz[-2::-2], liz[-1::-2])):
    print i,W_i,b_i


