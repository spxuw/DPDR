import torch
import numpy as np
import sys, copy, math, time, pdb
import os.path
import random
import pdb
import csv
import argparse
import itertools
from itertools import permutations, product
import itertools
from concurrent.futures import ThreadPoolExecutor


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

def get_batch(ztrn,ptrn,Q1,mb_size):
    s = torch.from_numpy(np.random.choice(np.arange(ptrn.size(dim=0), dtype=np.int64), mb_size, replace=False))
    batch_p = ztrn[s,:]
    batch_q = ptrn[s,:]
    batch_d = Q1[s,:]
    batch_t = t[:batch_time]
    return batch_p.to(device), batch_q.to(device),batch_d.to(device),batch_t.to(device)


def loss_bc(p_i,q_i):
    return torch.sum(torch.abs(p_i-q_i))/torch.sum(torch.abs(p_i+q_i))

def mean_loss(p_i, q_i):
    # Expand dimensions of p_i to match the shape of q_i
    expanded_p_i = p_i.expand_as(q_i)
    
    # Calculate the loss for each pair
    numerator = torch.sum(torch.abs(expanded_p_i - q_i), dim=1)
    denominator = torch.sum(torch.abs(expanded_p_i + q_i), dim=1)

    # Compute the mean loss
    return torch.mean(numerator / denominator).item()

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

def compute_for_index(i, Q2, ztst, ptst1, best_model, ptst2):
    print(f"Processing index: {i}")
    # Hyperparameters
    initial_temperature = 1000.0  # Increased initial temperature
    cooling_rate = 0.9999  # Faster cooling rate
    num_iterations = 200000
    patience = 2000  # Adaptive stopping criterion

    func1 = copy.deepcopy(best_model)

    # Initial setup
    current_solution = Q2[i]
    best_solution = Q2[i]
    current_loss = loss_bc(ptst2[i], ptst1[i])
    best_loss = current_loss
    best_comp = ptst2[i].clone()

    temperature = initial_temperature
    kk = 0
    no_improvement_steps = 0  # Track steps without improvement

    while kk < num_iterations:
        # Create a new solution by adding a dynamic perturbation
        new_solution = current_solution.clone()
        perturbation = np.random.normal(loc=0.1, scale=0.5, size=N_f)  # Perturbation with mean shift
        variation_scale_1 = 0.01 * (temperature / initial_temperature)  # Dynamic variation scale
        new_solution = new_solution + torch.from_numpy(variation_scale_1 * perturbation) * torch.sum(new_solution)
    
        # Enforce constraints on new solution
        new_solution[new_solution < 0] = 0
        if torch.sum(new_solution) == 0:
            new_solution = current_solution.clone()  # Reset if invalid
        new_solution = new_solution / torch.sum(new_solution)

        # Calculate distance and loss
        dis_nutrient = loss_bc(new_solution / torch.sum(new_solution), Q2[i] / torch.sum(Q2[i]))
        if dis_nutrient.item() < 1:
            new_composition = func1(torch.cat((ztst[i], new_solution), 0).unsqueeze(dim=0)).to(device)
            new_loss = loss_bc(new_composition, ptst1[i])

            # Accept new solution based on loss and temperature
            if new_loss.item() < current_loss.item() or np.random.rand() < np.exp((current_loss.item() - new_loss.item()) / temperature):
                current_solution = new_solution.clone()
                current_loss = new_loss

            # Update best solution if found
            if new_loss < best_loss:
                best_solution = new_solution.clone()
                best_loss = new_loss
                best_comp = new_composition.clone()
                no_improvement_steps = 0  # Reset no improvement steps counter
            else:
                no_improvement_steps += 1  # Increment if no improvement

        # Cooling and iteration update
        temperature *= cooling_rate
        kk += 1

        # Restart mechanism every 5000 iterations
        if kk % 5000 == 0:
            current_solution = Q2[i].clone()  # Restart from the initial point
            temperature = initial_temperature

    return best_solution.numpy(), best_comp.detach().numpy()

def train_reptile(max_epochs,mb,LR,zll,pall,ztrn,ptrn,ztst,ptst,zval,pval,Qll,Q1,Q2,Q3,ptar):

    ##################################### using PPN to learn map (q,z)-->p #####################
    loss_train = []
    loss_val = []
    qtst = np.zeros((ztst.size(dim=0),N_s))
    qtrn = np.zeros((zall.size(dim=0),N_s))

    func = ODEFunc().to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR)

    Loss_opt = 1
    for e in range(max_epochs):
        optimizer.zero_grad()
        batch_p, batch_q, batch_d, batch_t = get_batch(ztrn,ptrn,Q1,mb)
        
        # loss of the traning set
        for i in range(mb):
            p_pred = func(torch.cat((batch_p[i],batch_d[i]),0).unsqueeze(dim=0)).to(device)
            p_pred = torch.reshape(p_pred,(1,N_s))
            if i==0:
                loss = loss_bc(p_pred.unsqueeze(dim=0),batch_q[i].unsqueeze(dim=0))
            else:
                loss = loss + loss_bc(p_pred.unsqueeze(dim=0),batch_q[i].unsqueeze(dim=0))
        loss_train.append(loss.item()/mb)


        # validation set
        for i in range(zval.size(dim=0)):
            p_pred = func(torch.cat((zval[i],Q3[i]),0).unsqueeze(dim=0)).to(device)
            p_pred = torch.reshape(p_pred,(1,N_s))
            if i==0:
                l_val = loss_bc(p_pred.unsqueeze(dim=0),pval[i].unsqueeze(dim=0))
            else:
                l_val = l_val + loss_bc(p_pred.unsqueeze(dim=0),pval[i].unsqueeze(dim=0))
        loss_val.append(l_val.item()/zval.size(dim=0))
        if l_val.item()/zval.size(dim=0)<=Loss_opt:
            Loss_opt = loss_val[-1]
            best_model = copy.deepcopy(func)
        #print('epoch = ',e, 'loss = ', l_val.item()/mb)

        # update the neural network
        func.zero_grad()
        loss.backward()
        optimizer.step()

        if e == max_epochs-1:
            func = copy.deepcopy(best_model)
            if len(ztst.size())==2:
                for i in range(ztst.size(dim=0)):
                    pred_test = func(torch.cat((ztst[i],Q2[i]),0).unsqueeze(dim=0)).to(device)
                    pred_test = pred_test
                    pred_test = torch.reshape(pred_test,(1,N_s))
                    qtst[i,:] = pred_test.detach().numpy()
                for i in range(zall.size(dim=0)):
                    pred_test = func(torch.cat((zall[i],Qall[i]),0).unsqueeze(dim=0)).to(device)
                    pred_test = pred_test
                    pred_test = torch.reshape(pred_test,(1,N_s))
                    qtrn[i,:] = pred_test.detach().numpy()


    ################################ use simulated anneling to recommed #######################
    rtst = np.zeros((ztst.size(dim=0),N_f))
    dtst = np.zeros((ztst.size(dim=0),N_s))

    for i in range(200):
        r1, d1 = compute_for_index(i, Q2, ztst, ptar, best_model, ptst)
        rtst[i,] = r1
        dtst[i,] = d1 
    #for i, future in enumerate(futures):
    #    rtst[i,], dtst[i,] = future.result()
    return loss_train[-5:-1],qtst,qtrn,rtst,dtst


# hyperparameters
max_epochs = 1000
device = 'cpu'
batch_time = 100
t = torch.arange(0.0, 100.0, 0.01)

# load the dataset
filepath_ptrain = '../data/simulation/recomm/p_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ptest = '../data/simulation/recomm/p_disease_perm_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_qtrain = '../data/simulation/recomm/q_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_qtest = '../data/simulation/recomm/q_disease_perm_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ptarget = '../data/simulation/recomm/p_disease_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ztrain = '../data/simulation/recomm/z_healthy_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'
filepath_ztest = '../data/simulation/recomm/z_disease_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'_'+str(fold)+'.csv'

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

P3 = np.loadtxt(filepath_ptarget,delimiter=',')
P3[P3<threshold] = 0
ptar = process_data(P3)

# pre training to select the parameter
LR = 0.01
mb = 20


#pdb.set_trace()

loss_train,qtst,qtrn,rtst,dtst = train_reptile(max_epochs,mb,LR,zall,pall,ztrn,ptrn,ztst,ptst,zval,pval,Qall,Q1,Q2,Q3,ptar)
np.savetxt('../results/simulation/community/qtst_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'.csv',qtst,delimiter=',')
np.savetxt('../results/simulation/community/qtrn_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'.csv',qtrn,delimiter=',')
np.savetxt('../results/simulation/community/rtst_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'.csv',rtst,delimiter=',')
np.savetxt('../results/simulation/community/dtst_'+str(perturbation)+'_'+str(sparsity)+'_'+str(connectivity)+'_'+str(noise)+'_'+str(ratio)+'.csv',dtst,delimiter=',')


