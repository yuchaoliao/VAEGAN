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
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torchvision.utils import save_image

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

#set a seed for same validation set
random_seed = 42
torch.manual_seed(random_seed)

# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 20
# Number of channels in the training input
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 32
# Size of feature maps in discriminator
ndf = 32
# Number of training epochs
num_epochs = 150
# Learning rate for optimizers
lr_G = 0.07
lr_D = 0.0001
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# input binary bit size
bit_size = 32
# calculate number of variables may include directives
temp = pd.read_csv(open(files[0],'r'))
number_of_variable = int(len(temp) / bit_size)  

#custom dataset from all filenames 
input_dataset = CustomDataset.CustomDataset(filenames = files, batch_size = batch_size, bit_size = bit_size)
input_dataset_all = CustomDataset.CustomDataset(filenames = files, batch_size = 1, bit_size = bit_size)
input_dataset_all_in_one = CustomDataset.CustomDataset(filenames = files, batch_size = len(files), bit_size = bit_size)

dataloader = torch.utils.data.DataLoader(input_dataset,batch_size = None, shuffle = True)
dataloader_all = torch.utils.data.DataLoader(input_dataset_all,batch_size = None, shuffle = True)
dataloader_all_in_one = torch.utils.data.DataLoader(input_dataset_all_in_one,batch_size = None, shuffle = True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
         
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, (2,4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, (2,4), 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc,(2,2),2, 0, bias=False),
            nn.Sigmoid()
        )
  
    def forward(self, input):
        output = self.network(input)
        return output
    
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.network = nn.Sequential(
            # nc * 20 * 32
            nn.Conv2d(nc, ndf, (2,2), 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf * 10 * 16
            nn.Conv2d(ndf, ndf * 2, (2,4),  2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf*2 * 6 * 8
            nn.Conv2d(ndf * 2, ndf * 4,(2,4),  2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # ndf*4 * 4 * 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.network(input) 
    
    

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
netG.apply(weights_init)

# Print the model
print(netG)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(16, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
G_losses_epoch = []
D_losses_epoch = []
MMD_scores = []
SSD_scores = []
PRD_scores = []
COSS_scores = []

G_loss_temp = 0
D_loss_temp = 0
iters = 0
loss_G_max = -1
loss_D_min = np.Inf 
MDD_score_min = np.Inf
FID_score_min = np.Inf
loss_G_min = np.Inf
SSD_score_min = np.Inf
PRD_score_min = np.Inf
COSS_score_max = -1

############################
# Starting Training
###########################
print("Starting Training Loop...")

for epoch in range(num_epochs):
    G_loss_temp = 0
    D_loss_temp = 0
    
    netD.train()
    netG.train()
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device,dtype=torch.float)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        noise=0.9*noise+0.1*torch.randn((noise.size()), device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_loss_temp += errG.item()
        D_loss_temp += errD.item()
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(fake)

        iters += 1
    
    G_losses_epoch.append(G_loss_temp)
    D_losses_epoch.append(D_loss_temp)
    
    ############################
    # Evaluation process
    ###########################
    netG.eval()
    MMD_score = 0.0
    SSD_score = 0.0
    PRD_score = 0.0
    COSS_score = 0.0
    j = 0
    
    # Generate new fake data
    fixed_noise_test = torch.randn(len(files), nz, 1, 1, device=device)
    fake = netG(fixed_noise_test).detach().cpu()
    dataloader_fake = torch.utils.data.DataLoader(torch.round(fake),batch_size = batch_size, shuffle = False)
    dataloader_iterator = iter(dataloader_fake)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            fake_data = next(dataloader_iterator)
            
            # Change float to int 
            for index in range(20):
                if index != 0 and index != 11:
                    fake_data[:,:,index,20:] = 0
                    
            data = data.to(device,dtype=torch.float)
            data = data.view(data.size(0), -1)
            fake_data = torch.round(fake_data.view(fake_data.size(0), -1).to(device,dtype=torch.float))
            
            # Calculate and sum each metric for each mini-batch 
            if data.size(0) == batch_size and fake_data.size(0) == batch_size:
                j +=1
                MMD_score += Metrics.mmd_rbf(data, fake_data)
                SSD_score += Metrics.SSD(data,fake_data)
                PRD_score += Metrics.PRD(data,fake_data)
                COSS_score += Metrics.COSS(data,fake_data)
                if j == len(dataloader.dataset) - 2:
                    b_size = data.size(0)
                    num_rows = 8
                    both = torch.cat((data.view(b_size, 1, 20, 32)[:8], 
                                      fake_data.view(b_size, 1, 20, 32)[:8]))
                    save_image(both.cpu(), f"./Outputs/GAN_DCGAN_part1_20_output{epoch}.png", nrow=num_rows)
                    
        # Calculate average metrics per epoch
        MMD_epoch = MMD_score/j
        SSD_epoch = torch.sum(SSD_score/j)/batch_size
        PRD_epoch = torch.sum(PRD_score/j)/batch_size
        COSS_epoch = torch.sum(COSS_score/j)/batch_size

    # Save for plot
    MMD_scores.append(MMD_epoch.cpu().numpy())
    SSD_scores.append(SSD_epoch.cpu().numpy())
    PRD_scores.append(PRD_epoch.cpu().numpy())
    COSS_scores.append(COSS_epoch.cpu().numpy())
    
    # Save model with best MMD score
    if epoch != 0 and epoch != 1 and MMD_epoch < MDD_score_min:
        print('MDD_score decreased ({:.4f} --> {:.4f}). SSD_score decreased ({:.4f} --> {:.4f}). \n PRD_score decreased ({:.4f} --> {:.4f}). COSS_score increased ({:.4f} --> {:.4f}).Saving model ...'.format(
        MDD_score_min, MMD_epoch,SSD_score_min, SSD_epoch,PRD_score_min,  PRD_epoch, COSS_score_max,COSS_epoch ))
        torch.save(netG.state_dict(), 'GAN_DCGAN_netG_part1_20_32_test.pt')
        MDD_score_min = MMD_epoch
        SSD_score_min = SSD_epoch
        PRD_score_min = PRD_epoch
        COSS_score_max = COSS_epoch
    print('MDD_score: {:.4f} \t SSD_score: {:.4f} \t PRD_score: {:.4f} \t COSS_score: {:.4f}'.format(MMD_epoch,SSD_epoch,PRD_epoch,COSS_epoch))


############################
# Evaluate the model 
###########################
netG.load_state_dict(torch.load('GAN_DCGAN_netG_part1_20_32.pt'))

fixed_noise_test = torch.randn(len(files), nz, 1, 1, device=device)
fake = netG(fixed_noise_test).detach().cpu()

dataloader_fake = torch.utils.data.DataLoader(torch.round(fake),batch_size = batch_size, shuffle = False)
dataloader_iterator = iter(dataloader_fake)

MMD_score = 0.0
SSD_score = 0.0
PRD_score = 0.0
COSS_score = 0.0
j = 0
k = 5

for i, (data) in enumerate(dataloader):
    fake_data = next(dataloader_iterator)
    fake_data = torch.round(fake_data).to(device,dtype=torch.float)
    
    #change float to int 
    for index in range(20):
        if index != 0 and index != 11:
            fake_data[:,:,index,20:] = 0
            
    data = data.to(device,dtype=torch.float)
    data = data.view(data.size(0), -1)
    fake_data = fake_data.view(fake_data.size(0), -1)
     
     # Calculate and sum each metric for each mini-batch 
    if data.shape == fake_data.shape:
        j += 1
        MMD_score += Metrics.mmd_rbf(data, fake_data)
        SSD_score += Metrics.SSD(data,fake_data)
        PRD_score += Metrics.PRD(data,fake_data)
        COSS_score += Metrics.COSS(data,fake_data)
        
        # if k > 0:
        #     plt.imshow(data.view((batch_size,1,number_of_variable,bit_size))[0][0].cpu(),cmap="gray", vmin=0, vmax=1)
        #     plt.show()
        #     plt.imshow(fake_data.view((batch_size,1,number_of_variable,bit_size))[0][0].cpu(),cmap="gray", vmin=0, vmax=1)
            
        #     plt.show()
        #     k -=1
    
MDDscore = MMD_score.cpu().numpy()/j
SSDscore = sum(SSD_score.cpu().numpy()/j)/batch_size
PRDscore = sum(PRD_score.cpu().numpy()/j)/batch_size
COSSscore = sum(COSS_score.cpu().numpy()/j)/batch_size

print(f"MDDscore: {MDDscore}")
print(f"SSDscore: {SSDscore}")
print(f"PRDscore: {PRDscore}")
print(f"COSSscore: {COSSscore}")

############################
# Plot and save metrics
###########################
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.plot(G_losses_epoch, '-bx')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['G'])
plt.title('Loss vs. No. of epochs')

ax = f.add_subplot(1, 2, 2)
ax.yaxis.tick_right()
plt.plot(D_losses_epoch, '-rx')
plt.xlabel("Epoch")
plt.legend(['D'])
plt.title('Loss vs. No. of epochs')
plt.show()


df1 = pd.DataFrame(MMD_scores)
df2 = pd.DataFrame(SSD_scores)
df3 = pd.DataFrame(PRD_scores)
df4 = pd.DataFrame(COSS_scores)

# Concatenate dataframes
df = pd.concat([df1, df2,df3,df4], axis=1)

# Save to csv
df.to_csv('/Outputs/GAN_DCGAN_netG_part1_20_32_metrics.csv', index=False)

plt.plot(MMD_scores[1:150], '-rx')
plt.xlabel("Epoch")
plt.ylabel("MDD")
plt.legend(['MDD'])
plt.savefig('./Outputs/Figure/GAN_DCGAN_netG_part1_20_32_MMD.png')
plt.show()

plt.plot(SSD_scores[1:150], '-bx')
plt.xlabel("Epoch")
plt.ylabel("SSD")
plt.legend(['SSD'])
plt.savefig('./Outputs/Figure/GAN_DCGAN_netG_part1_20_32_SSD.png')
plt.show()

plt.plot(PRD_scores[1:150], '-gx')
plt.xlabel("Epoch")
plt.ylabel("PRD")
plt.legend(['PRD'])
plt.savefig('./Outputs/Figure/GAN_DCGAN_netG_part1_20_32_PRD.png')
plt.show()


plt.plot(COSS_scores[1:150], '-yx')
plt.xlabel("Epoch")
plt.ylabel("COSS")
plt.legend(['COSS'])
plt.savefig('./Outputs/Figure/GAN_DCGAN_netG_part1_20_32_COSSt.png')
plt.show()

# Plot loss
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.plot(G_losses, '-bx')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend(['G'])
plt.title('Loss vs. No. of minibatchs')

ax = f.add_subplot(1, 2, 2)
ax.yaxis.tick_right()
plt.plot(D_losses, '-rx')
plt.xlabel("Batch")
plt.legend(['D'])
plt.title('Loss vs. No. of minibatchs')
plt.show()

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()