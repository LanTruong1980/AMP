#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This Python code is for the submitted manuscript:
# The fundamental limits of sparse linear regression with sublinear sparsity


# In[1]:


import math
import numpy as np
from scipy import integrate
from numpy import linalg as LA
import matplotlib.pyplot as plt


# In[2]:


n=200
alpha=0.5
beta=0.9
m=math.floor(alpha*(n**beta))
k=n**beta
snrdB=np.array(range(0,5,1))
snr=10**(0.1*snrdB)
Deltavec=1/snr


# In[3]:


# Draw the theoretical limit (MMSE)
gap=10**(-4)  
step=gap/(n**(1-beta))
Evec=np.arange(0,1,step) # Trace tilE(Delta)


# In[4]:


tilE=np.zeros(len(Deltavec))
count=0
while count<len(Deltavec):
 Delta=Deltavec[count]
 E=Evec[0]
 pphi=(alpha/2)*(np.log(1+E/Delta)-E/(E+Delta))
 Sigma=np.sqrt((E+Delta)/(alpha*(n**(beta-1)))) 
 positivepart=lambda y: ((k/n)*(1/(Sigma*np.sqrt(2*math.pi)))*np.exp(-y**2/(2*Sigma**2))
                         + (1-k/n)*(1/(2*Sigma*np.sqrt(2*math.pi)))
                         *(np.exp(-(y-1)**2/(2*Sigma**2))
                         +np.exp(-(y+1)**2/(2*Sigma**2)))
                         *(np.log(1/(2*Sigma*np.sqrt(2*math.pi)))-(1/(2*Sigma**2))*(y+1)**2 
                         + 2*y/Sigma**2 + np.log((2*k/n)*np.exp((1-2*y)/(2*Sigma**2))+(1-k/n)*(1+np.exp(-2*y/Sigma**2))))
 iden1,err1=integrate.quad(positivepart, 0, 400)
 negativepart=lambda y: ((k/n)*(1/(Sigma*np.sqrt(2*math.pi)))*np.exp(-y**2/(2*Sigma**2))
                         + (1-k/n)*(1/(2*Sigma*np.sqrt(2*math.pi)))
                         *(np.exp(-(y-1)**2/(2*Sigma**2)) 
                         +np.exp(-(y+1)**2/(2*Sigma**2)))
                         *(np.log(1/(2*Sigma*np.sqrt(2*math.pi)))-(1/(2*Sigma**2))*(y-1)**2 
                         -2*y/Sigma**2+ np.log((2*k/n)*np.exp((1+2*y)/(2*Sigma**2)) 
                         +(1-k/n)*(1+np.exp(2*y/Sigma**2))))
 iden2,err2=integrate.quad(negativepart, -400, 0)
 iden=-(iden1+iden2)-(1/2)*np.log(2*math.pi*np.exp(1)*Sigma**2)
 fRSmin=pphi+n**(1-beta)*iden
 tilEtemp=E   
 for t in range(1,len(Evec)-1,1):
  E=Evec[t]
  pphi=(alpha/2)*(np.log(1+E/Delta)-E/(E+Delta))
  Sigma=np.sqrt((E+Delta)/(alpha*(n**(beta-1)))) 
  positivepart=lambda y: ((k/n)*(1/(Sigma*np.sqrt(2*math.pi)))*np.exp(-y**2/(2*Sigma**2)) 
                          + (1-k/n)*(1/(2*Sigma*np.sqrt(2*math.pi)))
                          *(np.exp(-(y-1)**2/(2*Sigma**2))
                          +np.exp(-(y+1)**2/(2*Sigma**2))))
                          *(np.log(1/(2*Sigma*np.sqrt(2*math.pi)))-(1/(2*Sigma**2))*(y+1)**2 
                          + 2*y/Sigma**2 + np.log((2*k/n)*np.exp((1-2*y)/(2*Sigma**2))
                          +(1-k/n)*(1+np.exp(-2*y/Sigma**2))))
  iden1,err1=integrate.quad(positivepart, 0, 400)
  negativepart=lambda y: ((k/n)*(1/(Sigma*np.sqrt(2*math.pi)))*np.exp(-y**2/(2*Sigma**2))
                          + (1-k/n)*(1/(2*Sigma*np.sqrt(2*math.pi)))
                          *(np.exp(-(y-1)**2/(2*Sigma**2))
                          +np.exp(-(y+1)**2/(2*Sigma**2))))
                          *(np.log(1/(2*Sigma*np.sqrt(2*math.pi)))-(1/(2*Sigma**2))*(y-1)**2 
                          -2*y/Sigma**2+ np.log((2*k/n)*np.exp((1+2*y)/(2*Sigma**2))
                          +(1-k/n)*(1+np.exp(2*y/Sigma**2))))
  iden2,err2=integrate.quad(negativepart, -400, 0)
  iden=-(iden1+iden2)-(1/2)*np.log(2*math.pi*np.exp(1)*Sigma**2)
  fRS=pphi+n**(1-beta)*iden
  if fRS<fRSmin:
    tilEtemp=E
    fRSmin=fRS
 tilE[count]=(n**(1-beta))*tilEtemp
 count+=1


# In[6]:


def etafunanderi(x,k,n,tau,Delta):
 tau=tau*np.sqrt(Delta)
 y=((1/2)*(1-k/n)*(np.exp(x/tau**2)-np.exp(-x/tau**2)))/((k/n)*np.exp(1/(2*tau**2))
                            +(1/2)*(1-k/n)*(np.exp(x/tau**2)+np.exp(-x/tau**2)))
 a=(1/(2*tau**2))*(1-k/n)*(k/n)*np.exp(1/(2*tau**2))*(np.exp(x/tau**2)+np.exp(-x/tau**2))+(1-k/n)**2
 b=((k/n)*np.exp(1/(2*tau**2))+(1/2)*(1-k/n)*(np.exp(x/tau**2)+np.exp(-x/tau**2)))**2
 yderitemp=a/b
 yderi=np.mean(yderitemp)
 return y,yderi

# Run AMP (iter=10)
alpha=alpha*n**(beta-1) #delta in the paper
MSE10=np.zeros(len(Deltavec))
for count in range(len(Deltavec)):
 Delta=Deltavec[count]
 maxouter=1000
 MSEouter10=np.zeros(maxouter)
 out=0 
 while out<maxouter:
 # Generate S,A,y
  s=np.zeros((n,1))
  for i in range(n):
   s[i]= np.random.binomial(1, 1-k/n)
   if s[i]==1:
    s[i]=2*np.random.binomial(1, 1/2)-1
  A=np.sqrt(1/n*beta)*np.random.normal(0,1,size=(m,n))
  W=np.random.normal(0,1,size=(m,1))
  y=np.matmul(A,s)+ W*np.sqrt(Delta)
  xhat=np.zeros((n,1))
  iterAMP=10
  z=np.zeros((m,1))
  iter=0
  pderi=0
  tau=np.sqrt(Delta+n**(1-beta)/alpha)
  while iter<iterAMP:
    z=y-np.matmul(A,xhat)+(n**(1-beta)/alpha)*z*pderi
    h=np.matmul(A.transpose(),z)+xhat
    p,pderi=etafunanderi(h,k,n,tau,n**(1-beta))
    xhat=p
    maxiter=5000
    u=np.random.normal(size=(maxiter,1)) 
    ma,era=etafunanderi(tau*u,k,n,tau,n**(1-beta))
    mb,erb=etafunanderi(1+tau*u,k,n,tau,n**(1-beta))
    mc,erc=etafunanderi(-1+tau*u,k,n,tau,n**(1-beta))
    v=(k/n)*ma**2+(1-k/n)*(1/2)*((mb-1)**2+(mc+1)**2)
    G=np.mean(v)
    tau=np.sqrt(Delta+(n**(1-beta)/alpha)*G)
    iter+=1
  MSEouter10[out]=(LA.norm(xhat-s,2))**2/n**(beta)
  out+=1 #outer
  MSE10[count]=np.mean(MSEouter10)   


# In[7]:


fig, ax = plt.subplots()
ax.plot(1/Deltavec,tilE,'b-o',1/Deltavec,MSE10,'m:x',linewidth=2)
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('MSE')
plt.legend(('Fundamental Limit','AMP (10 iterations)'))
ax.set_xlim((1, 2.5))
ax.set_ylim((0, 1))
ax.grid(True)
plt.savefig("ICML2021.pdf")

