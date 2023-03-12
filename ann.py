import numpy as np
import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

class TrafficANN(nn.Module):
    def __init__(self,d_in,h_1,h_2) -> None:
        super(TrafficANN,self).__init__()
        self.fc1 = nn.Linear(d_in,h_1)
        self.fc2 = nn.Linear(h_1,h_2)
        self.fc_out = nn.Linear(h_2,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        f = self.fc_out(x)
        return f
    
    def fit(self,X,y_data,criterion=nn.L1Loss(),n_steps=10000,batch_size=32,lr=1e-3,momentum=0.9,weight_decay=0,verbose=False):
        optimizer = optim.SGD(self.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        indices = np.arange(X.shape[0])
        for n in range(n_steps):
            optimizer.zero_grad()
            batch_idx = np.random.choice(indices,batch_size)
            X_batch = torch.from_numpy(X[batch_idx]).float()
            y_batch = torch.from_numpy(y_data[batch_idx]).unsqueeze(1).float()
            y_pred = self.forward(X_batch)
            loss = criterion(y_pred,y_batch)
            loss.backward()
            optimizer.step()
            if verbose and n % 1000 == 0:
                print(f'Loss at step {n}: {loss.item()}')

    def predict(self,X):
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            y_pred = self.forward(X)
            return y_pred.squeeze(1).numpy()