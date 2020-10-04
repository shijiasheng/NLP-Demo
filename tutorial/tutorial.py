import torch
import torch as F
import torch.nn.functional as E
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x=torch.linspace(-5,5,200) # x data (tenser), shape=(100,1)
x=Variable(x)
x_np=x.data.numpy()

y_relu=F.relu(x).data.numpy()
y_sigmoid=F.sigmoid(x).data.numpy()
y_tanh=F.tanh(x).data.numpy()
y_softplus=E.softplus(x).data.numpy()

#画图
plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np,y_sigmoid,c='red',label='sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np,y_tanh,c='red',label='tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np,y_softplus,c='red',label='softplus')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.show()