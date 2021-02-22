import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#only need to run once
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0 ,DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label),desc="Making trainin data"):
                try:
                    path = os.path.join(label,f)
                    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                    if label ==  self.CATS:
                        self.catcount += 1
                    if label ==  self.DOGS:
                        self.dogcount += 1
                except: #Exception as e:
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy",self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ",self.dogcount)
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

#data build, this is a loading test
training_data =np.load("training_data.npy",allow_pickle = True)
'''
print(len(training_data))
print(training_data[10])
plt.imshow(training_data[10][0],cmap="gray")
print(training_data[10][1])
plt.show()
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #input, output, kernel size
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        
        
        self.fc1 = nn.Linear(512,512) #see notes for the first 512
        self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        x =x.view(-1,512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim = 1)
net =Net()

optimizer = optim.Adam(net.parameters(),lr = 0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = .1
val_size = int(len(X)*VAL_PCT)
print("Validation Test Size: ", val_size)
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

BATCH_SIZE = 10
EPOCHS = 10

for epochs  in range(EPOCHS):
    for i in tqdm(range(0,len(train_X),BATCH_SIZE),desc="Training...."):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        #if using optimizer with 1 net.parameters()
        #optimizer.zero_grad() is the same thing
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs,batch_y)
        loss.backward()
        optimizer.step()
print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out =  net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ",round(correct/total,3))