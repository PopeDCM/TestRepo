#[0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1]
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx

start = np.array([[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[1],[0],[0],[1],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[1],[0],[0],[1],[0],[0],[0],[1],[1],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[1],[0],[0],[1]])
#j = np.reshape(start, (21,1))

np.random.seed(1)

n = 84
m = 40
mulMat = np.random.normal(0, 1, size=(m, n))
repr = np.matmul(mulMat, start)
repr2 = repr.reshape(m)


print(repr2)

vx = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [mulMat@vx == repr2] #, vx <= 1, vx>=0
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

print(vx.value)

solution = vx.value

index = 0
flag = "Success"
for i in solution:
    if i < .8 and start[index][0] == 1:
        flag = "Fail"
        break
    elif i > .8 and start[index][0] == 0:
        flag = "Fail"
        break
    index+=1



print(flag)
