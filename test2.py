import numpy as np,matplotlib.pyplot as plt
import torch
import torch.nn as nn,torch.nn.functional as F

# theta = torch.randn((100,1),requires_grad=True)
x = torch.randn((1,1),requires_grad=True)

f = lambda x: F.sigmoid(x**2)

def gamma (proba,u):
    if 0.5-(proba+u)>0:
        return 0
    else: return 1

u = 0
learning_rate =0.1

F_=[]
G=[]
U=[]
total_iter=None
initial_input = f(x)

for i in range(10000):
    # print (' f(x):%.2f, gamma:%.2f, u:%.2f'% (f(x).item(),gamma(f(x),u),u))
    F_.append(f(x).item())
    G.append(gamma(f(x).data,u))
    U.append(u)

    l = (f(x)-gamma(f(x).data,u)+u)**2
    l.backward()
    with torch.no_grad():
        x-=x.grad*learning_rate
    u = u + (f(x) - gamma(f(x).data,u))
    if np.abs((f(x) - gamma(f(x).data,u)).detach())<1e-8:
        total_iter=i+1
        break
print(initial_input.item(),total_iter)

    # print(theta.item(),f(theta).item(),gamma(f(theta),u),u.item())
plt.figure()

plt.plot(G,label='gamma')
plt.plot(U,label='error')
plt.plot(F_,label='f')
plt.legend()
plt.show()

