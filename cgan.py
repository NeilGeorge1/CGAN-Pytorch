import numpy as np  
import os 
import argparse
import torch 
from torch.autograd import Variable
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torchvision import datasets 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.utils as vutils
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to custom dataset (preprocessed CBIS)')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='image size input')
parser.add_argument('--channels', type=int, default=1, help='number of channels in images (e.g. grayscale=1, RGB=3)')
parser.add_argument('--latentdim', type=int, default=100, help='size of latent vector')
parser.add_argument('--n_classes', type=int, default=2, help='number of classes in data set')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--lrate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for Adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for Adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--randomseed', type=int, help='seed')

opt = parser.parse_args()

img_shape = (opt.channels, opt.imageSize, opt.imageSize)

cuda = True if torch.cuda.is_available() else False 

os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None: 
	opt.randomseed = random.randint(1, 10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

# Custom dataset loader for CBIS
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for img_file in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_file))
                self.labels.append(int(label))  # Assuming folder name corresponds to class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # convert to grayscale (1 channel)
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Data transformation
transform = transforms.Compose([
    transforms.Resize(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # assuming grayscale, change for RGB if needed
])

dataset = CustomDataset(root_dir=opt.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

# Building generator
class Generator(nn.Module): 
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(opt.n_classes, opt.n_classes)
        self.depth = 128

        def init(input, output, normalize=True): 
            layers = [nn.Linear(input, output)]
            if normalize: 
                layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers 

        self.generator = nn.Sequential(
            *init(opt.latentdim + opt.n_classes, self.depth), 
            *init(self.depth, self.depth * 2), 
            *init(self.depth * 2, self.depth * 4), 
            *init(self.depth * 4, self.depth * 8),
            nn.Linear(self.depth * 8, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels): 
        gen_input = torch.cat((self.label_embed(labels), noise), -1)
        img = self.generator(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# Building discriminator
class Discriminator(nn.Module): 
    def __init__(self): 
        super(Discriminator, self).__init__()
        self.label_embed1 = nn.Embedding(opt.n_classes, opt.n_classes)
        self.dropout = 0.4 
        self.depth = 512

        def init(input, output, normalize=True): 
            layers = [nn.Linear(input, output)]
            if normalize: 
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers 

        self.discriminator = nn.Sequential(
            *init(opt.n_classes + int(np.prod(img_shape)), self.depth, normalize=False),
            *init(self.depth, self.depth), 
            *init(self.depth, self.depth),
            nn.Linear(self.depth, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels): 
        imgs = img.view(img.size(0), -1)
        inpu = torch.cat((imgs, self.label_embed1(labels)), -1)
        validity = self.discriminator(inpu)
        return validity 

# Weight initialization
def init_weights(m): 
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Building generator 
generator = Generator()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Building discriminator  
discriminator = Discriminator()
discriminator.apply(init_weights)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Loss function
a_loss = torch.nn.BCELoss()

# Labels 
real_label = 0.9
fake_label = 0.0

FT = torch.LongTensor
FT_a = torch.FloatTensor

if cuda: 
    generator.cuda()
    discriminator.cuda()
    a_loss.cuda()
    FT = torch.cuda.LongTensor
    FT_a = torch.cuda.FloatTensor

# Training 
for epoch in range(opt.epoch): 
    for i, (imgs, labels) in enumerate(dataloader): 
        batch_size = imgs.shape[0]

        # Convert img, labels into proper form 
        imgs = Variable(imgs.type(FT_a))
        labels = Variable(labels.type(FT))
    
        # Creating real and fake tensors of labels 
        reall = Variable(FT_a(batch_size, 1).fill_(real_label))
        f_label = Variable(FT_a(batch_size, 1).fill_(fake_label))

        # Initializing gradients
        gen_optimizer.zero_grad() 
        d_optimizer.zero_grad()

        #### TRAINING GENERATOR ####
        noise = Variable(FT_a(np.random.normal(0, 1, (batch_size, opt.latentdim))))
        gen_labels = Variable(FT(np.random.randint(0, opt.n_classes, batch_size)))
        
        gen_imgs = generator(noise, gen_labels)
        
        # Ability for discriminator to discern the real vs generated images 
        validity = discriminator(gen_imgs, gen_labels)
        
        # Generative loss function 
        g_loss = a_loss(validity, reall)

        # Backpropagate
        g_loss.backward()
        gen_optimizer.step()

        #### TRAINING DISCRIMINATOR ####
        d_optimizer.zero_grad()

        # Loss for real images and labels 
        validity_real = discriminator(imgs, labels)
        d_real_loss = a_loss(validity_real, reall)

        # Loss for fake images and labels 
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = a_loss(validity_fake, f_label)

        # Total discriminator loss 
        d_loss = 0.5 * (d_fake_loss + d_real_loss)

        # Calculate gradients
        d_loss.backward()
        d_optimizer.step()

        if i % 100 == 0: 
            vutils.save_image(gen_imgs, f'{opt.output}/real_samples.png', normalize=True)
            fake = generator(noise, gen_labels)
            vutils.save_image(fake.detach(), f'{opt.output}/fake_samples_epoch_{epoch:03d}.png', normalize=True)

    print(f"[Epoch: {epoch+1}/{opt.epoch}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Checkpoints 
    torch.save(generator.state_dict(), f'{opt.output}/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'{opt.output}/discriminator_epoch_{epoch}.pth')
