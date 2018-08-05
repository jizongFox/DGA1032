import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


u = torch.zeros((1),requires_grad=False).float()
f = torch.tensor([0.7],requires_grad=True).float()

gamma = torch.zeros((1),requires_grad=False)
gamma[0]=1

# loss = lambda (f,u,gamma): (F.sigmoid(f)-gamma+u)**2
learning_rate= 0.1
U =[]
F_=[]
sigmod_F=[]
loss = []
p=10

for i in range (10000):

    l =  u*(F.sigmoid(f)-gamma)+ p/2.0*(F.sigmoid(f)-gamma)**2
    l.backward()

    with torch.no_grad():
        f -= learning_rate * f.grad
        f.grad.zero_()
        gamma = ((f>0.5)*1).float()
        u = u + p*(F.sigmoid(f)-gamma)


    U.append(u.item())
    F_.append(f.item())
    sigmod_F.append(F.sigmoid(f).item())

    print(f.item(),F.sigmoid(f).item())


import matplotlib.pyplot as plt

plt.plot(U,label='U')
plt.plot(F_,label="F")
plt.plot(sigmod_F,label="sigmoid F")
# plt.plot([x+y for (x,y) in zip(sigmod_F,U)])
plt.legend()
plt.show()

#
#
#
# u=0
# f= -1
# gamma = 1
# sigmoid = lambda x: 1 / (1 + np.exp(-x))
#
# loss = lambda (f,u,gamma) : f+ (sigmoid(f)-gamma+u)**2
#
# learning_rate=1
#
# def gradient(f,u,gamma):
#     # delta = 0.0000000001
#     # return (loss((f+delta,u,gamma))-loss((f,u,gamma)))/delta
#     if f > gamma:
#         return 0.01
#     else: return -0.01
#
# U=[]
# F=[]
# for i in range (10000):
#     f = f - gradient(f,u,gamma) * learning_rate
#     # gamma = (f > 0.5) * 1
#     u = u + f-1
#     U.append(u)
#     F.append(f)
#     print(f,sigmoid(f),u,gamma)
#
# import matplotlib.pyplot as plt
# plt.plot(U,label='U')
# plt.plot(F,label="F")
# plt.legend()
# plt.show()


'''
import torch

dtype = torch.float
# device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H,  dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out,  dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in,  dtype=dtype)
y = torch.randn(N, D_out, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H,  dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
'''