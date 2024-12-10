# Import Python related required packages
import io
import os
import cv2
import gdown
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde, norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm
import pickle

#Import torch related packages
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


import pytorch_lightning as pl
from modules import *
class Encoder(nn.Module):
    
    def __init__(self,encoded_space_dim,dim1,dim2,num_layers,num_neur):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Linear(dim1 * dim2, num_neur))
        self.layers.append(nn.ReLU(True))
        for i in range(num_layers):
          self.layers.append(nn.Linear(num_neur,num_neur))
          self.layers.append(nn.ReLU(True))
        
        self.layers.append(nn.Linear(num_neur,encoded_space_dim))

        self.encoder = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self,encoded_space_dim,dim1,dim2,num_layers,num_neur):
        super().__init__()

        self.layers = []
        # self.decoder = nn.Sequential(
        self.layers.append(nn.Linear(encoded_space_dim, num_neur))
        self.layers.append(nn.ReLU(True))
        for i in range(num_layers):
          self.layers.append(nn.Linear(num_neur, num_neur))
          self.layers.append(nn.ReLU(True))
        
        self.layers.append(nn.Linear(num_neur, dim1 * dim2))
        self.decoder = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = self.decoder(x)
        return x

def testy():
  print("HELLO BRO!")

class DiffSet():
    def __init__(self, train, ds):

        self.ds = ds
        self.size = 72
        self.depth = 2

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        ds_item = self.ds[item][0]

        data = F.pad(ds_item, (2, 2, 1, 1))
        #data = pad(ds_item)
        
        data = (data * 2.0) - 1.0 # normalize to [-1, 1].

        return data

class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.t_range = t_range
        self.in_size = in_size

        self.unet = Unet(dim = 64, dim_mults = (1, 2, 4, 8), channels=img_depth)

    def forward(self, x, t):
        return self.unet(x, t)

    def beta(self, t):
        # Just a simple linear interpolation between beta_small and beta_large based on t
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        # Product of alphas from 0 to t
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        #print('BRUH')
        #print(type(batch))
        #print(len(batch))
        #print(batch.shape)

        # Get a random time step for each image in the batch
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        # Generate noise, one for each image in the batch
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        # Run the noisy images through the U-Net, to get the predicted noise
        e_hat = self.forward(noise_imgs, ts)
        # Calculate the loss, that is, the MSE between the predicted noise and the actual noise
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0
            # Get the predicted noise from the U-Net
            e_hat = self.forward(x, t.view(1).repeat(x.shape[0]))
            # Perform the denoising step to take the image from t to t-1
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):        
      loss = self.get_loss(batch, batch_idx)
      self.log("train/loss", loss)
      return loss

    def validation_step(self, batch, batch_idx):      
      loss = self.get_loss(batch, batch_idx)
      self.log("val/loss", loss)
      return

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
      return optimizer

### Training function
def fit(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train().to(device)
    decoder.train().to(device)
    train_loss = []
    for data,label in dataloader: 
        img = data
        img = img.view(img.size(0), -1).to(device)  
        label = label.to(device)
        latent = encoder(img)
        decoded_img = decoder(latent)
        loss = loss_fn(decoded_img, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Valid function
def val(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval().to(device)
    decoder.eval().to(device)
    with torch.no_grad(): 
        list_decoded_img = []
        list_img = []
        for  data, label in dataloader:
            img = data
            img = img.view(img.size(0), -1).to(device) 
            label = label.to(device)
            latent = encoder(img)
            decoded_img = decoder(latent)
            list_decoded_img.append(decoded_img.cpu())
            list_img.append(img.cpu())
        list_decoded_img = torch.cat(list_decoded_img)
        list_img = torch.cat(list_img) 
        val_loss = loss_fn(list_decoded_img, list_img)
    return val_loss.data


###test and plot outputs
def test(encoder,decoder,dataset,device,loss_fn,in_d1,in_d2,n=10):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
         decoded_img  = decoder(encoder(img))
         loss = loss_fn(decoded_img,img)
         print('For image {}, the loss = {}'.format(i,loss.data))
      plt.plot(img.cpu().reshape(in_d1,in_d2).numpy()[0],img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(in_d1,in_d2).numpy()[0],decoded_img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  

###test and plot outputs. Makes plots in log scale. This is only for NR plots
def test_log(encoder,decoder,dataset,device,loss_fn,in_d1,in_d2,n=10):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
         decoded_img  = decoder(encoder(img))
         loss = loss_fn(decoded_img,img)
         print('For image {}, the loss = {}'.format(i,loss.data))
      plt.plot(img.cpu().reshape(in_d1,in_d2).numpy()[0],img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      plt.xscale('log')
      plt.yscale('log')
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(in_d1,in_d2).numpy()[0],decoded_img.cpu().reshape(in_d1,in_d2).numpy()[1]) 
      plt.xscale('log')
      plt.yscale('log')
      if i == n//2:
         ax.set_title('Reconstructed images')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()  

###Get latent variables
def get_latent_variables(encoder, decoder, device, dataloader):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): 
        # Define the lists to store the original images, the recreated ones,
        # the latent variables and the corresponding labels
        list_img = []
        list_decoded_img = []
        list_latent = []
        list_labels = []

        for  data, label in dataloader:
            img = data
            img = img.view(img.size(0), -1).to(device) 
            # Encode and Decode data
            latent = encoder(img)
            decoded_img = decoder(latent)
            # Append the network output and the original image to the lists
            list_img.append(img.cpu())
            list_decoded_img.append(decoded_img.cpu())
            list_latent.append(latent.cpu())
            list_labels.append(label.cpu())
# Convert list into a torch.tensor
        t_img = torch.cat(list_img)
        t_decoded_img = torch.cat(list_decoded_img)
        t_latent = torch.cat(list_latent) 
        t_labels = torch.cat(list_labels)
    return t_img, t_decoded_img, t_latent, t_labels

def plot_ae_outputs(encoder,decoder,dataset,device,n=10):
    plt.figure(figsize=(26,5.5))
    for i in range(10):
      ax = plt.subplot(2,n,i+1)
      img,_ = dataset[i]
      #Notice that below i'm loading an image only, so it needs to be flatten
      #before entering the network
      img = torch.flatten(img).to(device)
      encoder.eval().to(device)
      decoder.eval().to(device)
      with torch.no_grad():
        decoded_img = decoder(encoder(img))
      plt.plot(img.cpu().reshape(2,128).numpy()[0],img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n) 
      plt.plot(decoded_img.cpu().reshape(2,128).numpy()[0],decoded_img.cpu().reshape(2,128).numpy()[1]) 
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  