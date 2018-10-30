import numpy as np
W = np.zeros((3, 4))
b = np.zeros(4)

liz = np.shape(W)[1]
print "liz"

liz = [11,1,22,2,33,3,44,4]


la = [num for num in liz if liz.index(num) % 2 ==1]
le = [num for num in liz if liz.index(num) % 2 ==0]

for w,b in zip(le,la):
    print w,b


