import torch
import numpy as np
import sys, copy, math, time, pdb
import os.path
import random
import pdb
import csv
import argparse
import itertools
import torch.optim as optim
from torchdiffeq import odeint
import itertools


parser = argparse.ArgumentParser(description='cnode')
parser.add_argument('--perturbation', default=None, help='sparsity')
parser.add_argument('--sparsity', default=None, help='sparsity')
parser.add_argument('--connectivity', default=None, help='connectivity')
parser.add_argument('--noise', default=None, help='noise')
parser.add_argument('--ratio', default=None, help='ratio')
parser.add_argument('--fold', default=None, help='fold')

args = parser.parse_args()
sparsity = args.sparsity
connectivity = args.connectivity
noise = args.noise
ratio = args.ratio
fold = args.fold
perturbation = args.perturbation

N_s = 100
N_f = 60
threshold = 1e-6
y_s_global = None 

def ensure_float(tensor):
    return tensor.float() if tensor.dtype != torch.float32 else tensor
    
def get_batch(ztrn,ptrn,Q1,mb_size):
    s = torch.from_numpy(np.random.choice(np.arange(ptrn.size(dim=0), dtype=np.int64), mb_size, replace=False))
    batch_p = ztrn[s,:]
    batch_q = ptrn[s,:]
    batch_d = Q1[s,:]
    batch_t = t[:batch_time]
    return batch_p.to(device), batch_q.to(device),batch_d.to(device),batch_t.to(device)


def loss_bc(p_i,q_i):
    return torch.sum(torch.abs(p_i-q_i))/torch.sum(torch.abs(p_i+q_i))

def mean_loss(p_i,q_i):
    loss_a = 0
    for i in range(q_i.size(dim=0)):
        loss_a = loss_a + loss_bc(p_i.unsqueeze(dim=0),q_i[i].unsqueeze(dim=0)).detach().numpy()
    return loss_a/q_i.size(dim=0)

def process_data(P):
    #Z = P.copy()
    #Z[Z>0] = 1
    P = P/P.sum(axis=0)[np.newaxis,:]
    #Z = Z/Z.sum(axis=0)[np.newaxis,:]
    
    P = P.astype(np.float32)
    #Z = Z.astype(np.float32)

    P = torch.from_numpy(P.T)
    #Z = torch.from_numpy(Z.T)
    return P


class ODEFunc(torch.nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.fcc1 = torch.nn.Linear(N_f, N_s)
        self.fcc2 = torch.nn.Linear(N_s, N_s)

    def forward(self, y):
        y1 = y[:,N_s:].float()
        y2 = y[:,:N_s].float()
        out = self.fcc1(y1.float())
        out = self.fcc2(out.float())
        return torch.mul(y2,torch.abs(out))/torch.sum(torch.mul(y2,torch.abs(out)))
        
def train_reptile(max_epochs,mb,LR,zll,pall,ztrn,ptrn,ztst,ptst,zval,pval,Qll,Q1,Q2,Q3):
    loss_train = []
    loss_val = []
    qtst = np.zeros((ztst.size(dim=0), N_s))
    qtrn = np.zeros((ztrn.size(dim=0), N_s))

    model = ODEFunc().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    Loss_opt = 1
    for e in range(max_epochs):
        optimizer.zero_grad()
        batch_p, batch_q, batch_d, batch_t = get_batch(ztrn, ptrn, Q1, mb)

        # Training loss
        loss = 0
        for i in range(mb):
            p_pred = model(torch.cat((batch_p[i], batch_d[i]), 0).unsqueeze(dim=0), batch_t).to(device)
            p_pred = torch.reshape(p_pred, (1, N_s))
            loss += loss_bc(p_pred.unsqueeze(dim=0), batch_q[i].unsqueeze(dim=0))
        loss_train.append(loss.item() / mb)

        # Validation loss
        l_val = 0
        for i in range(zval.size(dim=0)):
            p_pred = model(torch.cat((zval[i], Q3[i]), 0).unsqueeze(dim=0), batch_t).to(device)
            p_pred = torch.reshape(p_pred, (1, N_s))
            l_val += loss_bc(p_pred.unsqueeze(dim=0), pval[i].unsqueeze(dim=0))
        loss_val.append(l_val.item() / zval.size(dim=0))
        #print(l_val.item() / zval.size(dim=0))

        if l_val.item() / zval.size(dim=0) <= Loss_opt:
            Loss_opt = loss_val[-1]
            best_model = copy.deepcopy(model)

        # Update the neural network
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if e == max_epochs - 1:
            model = copy.deepcopy(best_model)
            if len(ztst.size()) == 2:
                for i in range(ztst.size(dim=0)):
                    pred_test = model(torch.cat((ztst[i], Q2[i]), 0).unsqueeze(dim=0), batch_t).to(device)
                    qtst[i, :] = pred_test.detach().numpy()
                for i in range(ztrn.size(dim=0)):
                    pred_test = model(torch.cat((ztrn[i], Q1[i]), 0).unsqueeze(dim=0), batch_t).to(device)
                    qtrn[i, :] = pred_test.detach().numpy()

    return loss_train[-5:-1],qtst,qtrn


# hyperparameters
max_epochs = 1000
device = 'cpu'
batch_time = 100
t = torch.arange(0.0, 100.0, 0.01)

# load the dataset
filepath_ptrain = '../data/simulation/mapping/p_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ptest = '../data/simulation/mapping/p_healthy_test_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_qtrain = '../data/simulation/mapping/q_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_qtest = '../data/simulation/mapping/q_healthy_test_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ztrain = '../data/simulation/mapping/z_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ztest = '../data/simulation/mapping/z_healthy_test_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'

# compostion and dietary of healthy
P1 = np.loadtxt(filepath_ptrain,delimiter=',')
Q1 = np.loadtxt(filepath_qtrain,delimiter=',')
Z1 = np.loadtxt(filepath_ztrain,delimiter=',')

number_of_cols = P1.shape[1]
random_indices = np.random.choice(number_of_cols, size=int(0.2*number_of_cols), replace=False)
P_val = P1[:,random_indices]
Z_val = Z1[:,random_indices]

Qall = Q1
Q3 = Q1[:,random_indices]
P_train =  P1[:,np.setdiff1d(range(0,number_of_cols),random_indices)]
Q1 = Q1[:,np.setdiff1d(range(0,number_of_cols),random_indices)]
Z_train = Z1[:,np.setdiff1d(range(0,number_of_cols),random_indices)]

P_train[P_train<threshold] = 0
P_val[P_val<threshold] = 0
P1[P1<threshold] = 0

pall = process_data(P1)
zall = process_data(Z1)
ptrn = process_data(P_train)
ztrn = process_data(Z_train)
pval = process_data(P_val)
zval = process_data(Z_val)
M, N = ptrn.shape

Qall = torch.from_numpy(Qall.T)
Q1 = torch.from_numpy(Q1.T)
Q3 = torch.from_numpy(Q3.T)

# compostion and dietary of diseased
P2 = np.loadtxt(filepath_ptest,delimiter=',')
Z2 = np.loadtxt(filepath_ztest,delimiter=',')
P2[P2<threshold] = 0
ptst = process_data(P2)
ztst = process_data(Z2)

Q2 = np.loadtxt(filepath_qtest,delimiter=',')
Q2 = torch.from_numpy(Q2.T)

# pre training to select the parameter
LR = 0.01
mb = 20


loss_train,qtst,qtrn = train_reptile(max_epochs,mb,LR,zall,pall,ztrn,ptrn,ztst,ptst,zval,pval,Qall,Q1,Q2,Q3)
np.savetxt('../results/simulation_ode/mapping/qtst_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv',qtst,delimiter=',')
np.savetxt('../results/simulation_ode/mapping/qtrn_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv',qtrn,delimiter=',')


