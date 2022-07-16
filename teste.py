import os
import random
import zipfile
import seaborn as sns
import torch
from torch_snippets import *
from torchvision import models, transforms
from torchvision.transforms.transforms import ToPILImage
import json
from numba import jit, cuda

# Helpers

def buildPathFor(relative_path):
    script_dir = os.path.dirname(__file__) # Absolute dir the script is in
    return os.path.join(script_dir, relative_path)

# Permite a duplicação de partes da biblioteca
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print(torch.zeros(1).cuda) # Verifica se a GPU é compativel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Parte 1

class SiameseNetworkDataset(Dataset):
    def __init__(self, image_folder, imagenames_file, transform=None):
        self.image_folder = image_folder
        self.imagenames_file = imagenames_file
        self.nitems = 0
        self.transform = transform

        f = open(self.imagenames_file,"r")
        self.pairs = []

        for line in f:
            pair = line.strip().split(" ")
            self.pairs.append(pair)

        self.nitems = len(self.pairs)

    def __getitem__(self, ix):
        image1 = self.pairs[ix][0]
        image2 = self.pairs[ix][1]
        
        person1 = image1.split("_")[0] # 001_01.png -> 001
        person2 = image2.split("_")[0] # 002_01.png -> 002
        
        if (person1 == person2):
            truelabel = 0
        
        else:
            truelabel = 1
        
        image1 = read("{}/{}".format(self.image_folder,image1))
        image2 = read("{}/{}".format(self.image_folder,image2))
        image1 = np.expand_dims(image1,2)
        image2 = np.expand_dims(image2,2)
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, np.array([truelabel])

    def __len__(self):
        return self.nitems


# Regular preprocessing transformation. After being resized, 
# it is converted into a tensor for normalization.

# Such transformations are applied everytime images are loaded from the filename lists in training, validation, 
# and test sets. We will do that during training, then by adding affine transformations and increasing the number 
# of epochs, we are actually implementing data augmentation. 

prep = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(5, (0.01,0.2),
    #                         scale=(0.9,1.1)),
    transforms.Resize((100,100)),
    transforms.ToTensor()
    # transforms.Normalize((0.5), (0.5))
])

prepVGG = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAffine(5, (0.01,0.2),
    #                         scale=(0.9,1.1)),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    # transforms.Normalize((0.5), (0.5))
])

aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300,300), interpolation=transforms.InterpolationMode.BILINEAR, 
                      max_size=None, antialias=True),
    transforms.RandomAffine(degrees=15, translate=(0.05,0.10), scale=(0.9,1.1), shear=(-2,2),
                            interpolation=transforms.InterpolationMode.BILINEAR, 
                            fill=0, fillcolor=None, resample=None),
    transforms.CenterCrop(250),
    transforms.Resize((100,100), interpolation=transforms.InterpolationMode.BILINEAR, 
                      max_size=None, antialias=True),
    transforms.ToTensor()
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   
])

path = buildPathFor("datasets/comparisons_A.txt")
image_folder = buildPathFor("datasets/DB1_A") # folder with images of a dataset
train_imagenames_file = buildPathFor("datasets/content/train_comparisons.txt") # text file with image comparisons for training
valid_imagenames_file = buildPathFor("datasets/content/valid_comparisons.txt") # text file with image comparisons for validation

with open(path, 'r') as fp:
    data = [(random.random(), line) for line in fp]
    dataset_size = len(data)
    train_perc = 0.80
    valid_perc = 0.20
    num_train_samples = int(dataset_size*train_perc)
    num_valid_samples = int(dataset_size*valid_perc)
data.sort()

with open(train_imagenames_file, 'w') as train:
  for _, line in data[:num_train_samples]:
    train.write(line)
train.close()

with open(valid_imagenames_file, 'w') as valid:
  for _, line in data[num_train_samples:]:
    valid.write(line)
valid.close()

print(train)

# criação do objeto da classe dataset
dataset = SiameseNetworkDataset(image_folder = image_folder, imagenames_file = train_imagenames_file, transform = aug)

print("Number of images: {}".format(len(dataset)))

# obtendo uma amostra aleatória do conjunto de dados
sample = random.randint(0, len(dataset))
image1, image2, label = dataset[sample]

# Visualizando as duas imagens de entrada, bem como o label
nchannels = image1.shape[0]
height = image1.shape[1]
width = image1.shape[2]

print("Images are {}x{}x{}".format(width,height,nchannels))

image1 = image1.squeeze()
image2 = image2.squeeze()


# Plotting images
fig, ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].imshow(image1, cmap='gray')
ax[1].imshow(image2, cmap='gray')
dissimilarity_text = 'Different' if label[0] else 'Equal'
plt.suptitle('Dissimilarity: {} - {}'.format(label[0], dissimilarity_text))
plt.subplots_adjust(hspace=0.5)
plt.show()

# Parte 2

def convBlock(ni, no):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size=3, padding=1, bias=False), #, padding_mode='reflect'),
        nn.BatchNorm2d(no),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.features = nn.Sequential(
            convBlock(1,16),
            convBlock(16,128),
            nn.Flatten(),
            nn.Linear(128*25*25, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )
  
    def forward(self, input1, input2):
        output1 = self.features(input1)
        output2 = self.features(input2)
        return output1, output2


class SiameseNetworkVGGBackbone(nn.Module):
    def __init__(self):
        super(SiameseNetworkVGGBackbone, self).__init__()
        self.features = nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-1])
        for param in self.features.parameters():
          param.requires_grad = False
      
        self.dimensionality_reductor = nn.Sequential(
            nn.Flatten(),
            models.vgg16(pretrained=False).classifier[0],
            nn.Linear(4096, 512), nn.ReLU(inplace = True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )


    def forward(self, input1, input2):
        output1 = self.features(input1)
        output1 = self.dimensionality_reductor(output1)
        output2 = self.features(input2)
        output2 = self.dimensionality_reductor(output2)

        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)/2 +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))/2
        acc = ((euclidean_distance > contrastive_thres) == label).float().mean()
        return loss_contrastive, acc


def train_batch(model, data, optimizer, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    optimizer.zero_grad()
    codesA, codesB = model(imgsA, imgsB)
    loss, acc =     criterion(codesA, codesB, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

def valid_batch(model, data, criterion):
    imgsA, imgsB, labels = [t.to(device) for t in data]

    model.eval() # put the model in evaluation mode
    codesA, codesB = model(imgsA, imgsB) # predict the output for the batch
    loss, acc  = criterion(codesA, codesB, labels) # compute loss, both inputs must have the same sizes
    model.train()
    
    return loss.item(), acc.item()

# Parte 3

batchsize = 32

def GetBatches(image_folder, image_names, batchsize, transformation):
    datatensor = SiameseNetworkDataset(image_folder, image_names, transformation) 
    dataloader = DataLoader(datatensor, batch_size=batchsize, shuffle=True)
    return(dataloader)

# as transformations, you may choose None, prep, or aug. However, aug applies to the training set only
# trainload = GetBatches(image_folder, train_imagenames_file, batchsize, prep)  
# validload = GetBatches(image_folder, valid_imagenames_file, batchsize, prep)

trainload = GetBatches(image_folder, train_imagenames_file, batchsize, prepVGG)  # descomentar essa parte se for usar o backbone da VGG
validload = GetBatches(image_folder, valid_imagenames_file, batchsize, prepVGG)

print("Quantidade de batches de treino: ", len(trainload))
print("Quantidade de batches de validação: ", len(validload))

inspect(next(iter(trainload))) # inspect a couple of items in the batches

# model = SiameseNetwork().to(device)
model = SiameseNetworkVGGBackbone().to(device)

criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01)
nepochs = 200 # default training value was 200, but requires a lot of time

contrastive_thres = 1.1


nepochs = 33
log = Report(nepochs)
for epoch in range(nepochs):
    N = len(trainload)
    for i, data in enumerate(trainload):
        batch_loss, batch_acc = train_batch(model, data, optimizer, criterion)
        log.record(epoch+(1+i)/N, trn_loss = batch_loss, trn_acc = batch_acc, end='\r')

    N = len(validload)
    with torch.no_grad():
        for i, data in enumerate(validload):
            batch_loss, batch_acc = valid_batch(model, data, criterion)
            log.record(epoch+(1+i)/N, val_loss = batch_loss, val_acc = batch_acc, end='\r')  
log.plot_epochs()       

torch.save(model.state_dict(), "saved_model_state_dict.pth")

# Parte 4

# Se já possui um modelo treinado, carregue os pesos
# model_path = buildPathFor('datasets/content/saved_model_state_dict.pth')
model_path = "saved_model_state_dict.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

test_image_folder = buildPathFor("datasets/DB1_A") # folder with images of a dataset
test_imagenames_file = buildPathFor("datasets/comparisons_A.txt") # csv file with image comparisons for test
testload = GetBatches(test_image_folder, test_imagenames_file, batchsize, prepVGG)

# Parte 5 - Deploy

# Put model in evaluation mode
model.eval()

Acc = []
Loss = []
contrastive_thres = 0.9

# Process all batches
for ix, data in enumerate(testload):
    imgsA, imgsB, labels = [t.to(device) for t in data]
    with torch.no_grad():
      codesA, codesB = model(imgsA, imgsB)

      loss, acc = criterion(codesA, codesB, labels)
      Acc.append(acc.detach().cpu().numpy())
      Loss.append(loss.detach().cpu().numpy())

# Como o dataset é desbalanceado, talvez seja interessante propor e 
# explorar outras métricas que não sejam apenas a acurácia e a Loss para a validação!
print('Acurácia no conjunto de teste: {:.6f}'.format(np.mean(Acc)))
print('Loss no conjunto de teste: {:.6f}'.format(np.mean(Loss)))

 # Distance histogram
different = []
same = []
model.eval()

image_folder = buildPathFor('datasets/DB1_A')
imagenames_file = buildPathFor('datasets/comparisons_A.txt')

dataset = SiameseNetworkDataset(image_folder=image_folder, imagenames_file = imagenames_file, transform = prepVGG)

dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

for ix, data in enumerate(dataloader):
  if (ix + 1) % 1000 == 0:
    print('Processing batch {}/{}'.format((ix + 1), len(dataloader)))

  imgsA, imgsB, labels = [t.to(device) for t in data]
  codesA, codesB = model(imgsA, imgsB)

  with torch.no_grad():
    euclidean_distance = F.pairwise_distance(codesA, codesB)
    if (labels == 0): # same person
      same.append(euclidean_distance.item())
    
    else:
      different.append(euclidean_distance.item())

fig, ax = plt.subplots(figsize = (10,5))
ax.set_title('Histograma de distâncias para as duas classes', fontsize = 20, fontweight = 'bold')
ax.set_xlabel('Distância Euclidiana', fontsize = 16, fontweight = 'bold')
ax.set_ylabel('Densidade', fontsize = 16, fontweight = 'bold')
ax.hist(different,bins = 50, alpha = 0.7, label = 'different')
ax.hist(same, bins = 50, alpha = 0.7, label = 'same')
ax.tick_params(labelsize = 16, axis = 'both')
ax.legend()
ax.grid(True)
plt.plot()

model.eval()
do_comparison = 'y'

image_folder = buildPathFor('datasets/DB1_A')
imagenames_file = buildPathFor('datasets/comparisons_A.txt')

dataset = SiameseNetworkDataset(image_folder = image_folder, imagenames_file = imagenames_file, transform = prepVGG)

dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

while(do_comparison == 'y'):
    dataiter = iter(dataloader)
    image1, image2, truelabel = [t.to(device) for t in next(dataiter)]
    concatenated = torch.cat((image1 * 0.5 + 0.5, image2 * 0.5 + 0.5), 0)
    output1, output2 = model(image1, image2)
    euclidean_distance = F.pairwise_distance(output1, output2)
    
    if (euclidean_distance.item() <= contrastive_thres):
        if (truelabel != 0):
            output = 'Same Person, which is an error.'
        
        else:
            output = 'Same Person, which is correct.'
    
    else:
        if (truelabel == 0):
            output = 'Different, which is an error.'
        
        else:
            output = 'Different, which is correct.'
    
    show(torchvision.utils.make_grid(concatenated),
         title='Dissimilarity: {:.2f}\n{}'.format(euclidean_distance.item(), output))
        
    do_comparison = input("Type y to continue: ")
    plt.show()