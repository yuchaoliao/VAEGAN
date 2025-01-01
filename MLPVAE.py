import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset
import glob
import os
import argparse
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import functions
import Metrics
import CustomDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'Is CUDA supported by this system?{torch.cuda.is_available()}')

print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

# example getting the number of cpu cores
from multiprocessing import cpu_count
# get the number of logical cpu cores
n_cores = cpu_count()
# report the number of logical cpu cores
print(f'Number of Logical CPU cores: {n_cores}')

partname = "xc7v585tffg1157-3"
benchmark_names = ["syrk", "syr2k", "mvt", "k3mm", "k2mm","gesummv", "gemm", "bicg", "atax"]
i = 0
for benchmark_name in benchmark_names:
    print(benchmark_name)
    if i == 0 :
        files = glob.glob("./Inputs/20_polybench/" + benchmark_name + "/" + partname + "/*")
        i += 1
    else: 
        files += glob.glob("./Inputs/20_polybench/" + benchmark_name + "/" + partname + "/*")
print("Total number of files: ", len(files))
print("Showing first 10 files...")
files[:10]


transform = transforms.Compose([transforms.ToTensor()])

# Set a seed for same validation set
random_seed = 42
torch.manual_seed(random_seed)

# Number of subprocesses to use for data loading
num_workers = 4
# Samples per batch
batch_size = 20
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr = 0.0001
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Percentage of training set to use as validation
valid_size = 0.1
# latent space feature size
features = 16
# input binary bit size
bit_size = 32

temp = pd.read_csv(open(files[0],'r'))
number_of_variable = int(len(temp) / bit_size)  # calculate number of variables may include directives

#custom dataset from all filenames 
input_dataset = CustomDataset.CustomDataset(filenames = files, batch_size = batch_size, bit_size = bit_size)


num_train = len(input_dataset)
indices = list(range(num_train))
np.random.seed(random_seed)
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

dataloader = torch.utils.data.DataLoader(input_dataset,batch_size = None, shuffle = True)

input_dataset_all = CustomDataset.CustomDataset(filenames = files, batch_size = 1, bit_size = bit_size)
input_dataset_all_in_one = CustomDataset.CustomDataset(filenames = files, batch_size = len(files), bit_size = bit_size)

dataloader_all = torch.utils.data.DataLoader(input_dataset_all,batch_size = None, shuffle = True)
dataloader_all_in_one = torch.utils.data.DataLoader(input_dataset_all_in_one,batch_size = None, shuffle = True)

# Calculate the number of samples to include in each set
dataset_size = len(input_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.8 * dataset_size))

np.random.shuffle(indices)

# Create data samplers and loaders
train_indices, val_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(input_dataset, batch_size=None, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(input_dataset, batch_size=None, sampler=valid_sampler)

# Define MLP VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=number_of_variable*bit_size, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=256)
        self.dec2 = nn.Linear(in_features=256, out_features=number_of_variable*bit_size)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var
    

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, (data) in enumerate(dataloader):
        #data, _ = data
        data = data.to(device,dtype=torch.float)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    MMD_score = 0.0
    fid_score = 0.0
    SSD_score = 0.0
    PRD_score = 0.0
    COSS_score = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            #data, _ = data
            data = data.to(device,dtype=torch.float)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            
            reconstruction = torch.round(reconstruction)
            if data.size(0) == batch_size:
                MMD_score += Metrics.mmd_rbf(data, reconstruction)
                SSD_score += Metrics.SSD(data, reconstruction)
                PRD_score += Metrics.PRD(data, reconstruction)
                COSS_score += Metrics.COSS(data, reconstruction)
            # if  data.size(0) == batch_size:
            #     fid_score += calculate_fretchet(nn.functional.pad(data.view((batch_size,1,18,32)),(0, 0, 7, 7), "constant", 0).repeat(1, 3, 3, 3),
            #                                     nn.functional.pad(reconstruction.view((batch_size,1,18,32)),(0, 0, 7, 7), "constant", 0).repeat(1, 3, 3, 3))
            
            # Save the last batch input and output of every epoch
            if i == len(dataloader.dataset):
                data = torch.round(data)
                reconstruction = torch.round(reconstruction)
                b_size = data.size(0)
                num_rows = 8
                both = torch.cat((data.view(b_size, 1, 20, 32)[:8], 
                                  reconstruction.view(b_size, 1, 20, 32)[:8]))
                save_image(both.cpu(), f"./Outputs/VAE_part1_20_output{epoch}.png", nrow=num_rows)
                
    val_loss = running_loss/len(dataloader.dataset)
    MDD = MMD_score.cpu().numpy()/(len(dataloader.dataset)-1)
    # FID = fid_score/(len(dataloader.dataset)-1)
    SSDscore = torch.sum(SSD_score/(len(dataloader.dataset)-1)).cpu().numpy()/batch_size
    PRDscore = torch.sum(PRD_score/(len(dataloader.dataset)-1)).cpu().numpy()/batch_size
    COSSscore = torch.sum(COSS_score/(len(dataloader.dataset)-1)).cpu().numpy()/batch_size
    
    return val_loss, MDD, SSDscore, PRDscore, COSSscore

model = LinearVAE().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# Train
train_loss = []
val_loss = []
MMD_score = []
FID_score = []
SSD_score = []
PRD_score = []
COSS_score = []

for epoch in range(num_epochs):
    train_epoch_loss = fit(model, dataloader)
    val_epoch_loss, MMD_epoch, SSD_epoch ,PRD_epoch, COSS_epoch= validate(model, dataloader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    MMD_score.append(MMD_epoch)
    # FID_score.append(FID_epoch)
    SSD_score.append(SSD_epoch)
    PRD_score.append(PRD_epoch)
    COSS_score.append(COSS_epoch)
    print('[%d/%d] \tLoss_train: %.4f\tLoss_val: %.4f\tMMD: %.4f\tSSD: %.4f \tPRD: %.4f \tCOSS: %.4f '
          % (epoch, num_epochs, train_epoch_loss, val_epoch_loss, MMD_epoch,SSD_epoch,PRD_epoch,COSS_epoch))
    

# Plot and save    
df1 = pd.DataFrame(MMD_score)
df2 = pd.DataFrame(SSD_score)
df3 = pd.DataFrame(PRD_score)
df4 = pd.DataFrame(COSS_score)

# concatenate dataframes
df = pd.concat([df1, df2,df3,df4], axis=1)

# save to csv
df.to_csv('/Outputs/VAE_MLP_part1_20_32_metrics.csv', index=False)

torch.save(model.state_dict(), 'VAE_MLP_part1_20_32.pt')
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.plot(train_loss, '-bx')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train'])
plt.title('Loss vs. epochs')


ax = f.add_subplot(1, 2, 2)
ax.yaxis.tick_right()
plt.plot(val_loss, '-rx')
plt.xlabel("Epoch")
plt.legend(['Valid'])
plt.title('Loss vs. epochs')
plt.show()

plt.plot(MMD_score, '-rx')
plt.xlabel("Epoch")
plt.ylabel("MDD")
plt.legend(['MDD'])
plt.savefig('./Outputs/Figure/VAE_MLP_part1_20_32_MMD.png')
plt.show()

plt.plot(SSD_score, '-bx')
plt.xlabel("Epoch")
plt.ylabel("SSD")
plt.legend(['SSD'])
plt.savefig('./Outputs/Figure/VAE_MLP_part1_20_32_SSD.png')
plt.show()

plt.plot(PRD_score, '-gx')
plt.xlabel("Epoch")
plt.ylabel("PRD")
plt.legend(['PRD'])
plt.savefig('./Outputs/Figure/VAE_MLP_part1_20_32_PRD.png')
plt.show()


plt.plot(COSS_score, '-yx')
plt.xlabel("Epoch")
plt.ylabel("COSS")
plt.legend(['COSS'])
plt.savefig('./Outputs/Figure/VAE_MLP_part1_20_32_COSS.png')
plt.show()