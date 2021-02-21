import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST(" ",train=True,download=True,
transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST(" ",train=False,download=True,
transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)
'''
for data in trainset:
    print(data)
    break
x,y = data[0][0],data[1][0]
print(y)


plt.imshow(data[0][0].view(28,28))
plt.show()
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__() #this runs inherrited init from nn.module
        #input flattened image size, output whatever
        self.fc1 = nn.Linear(28*28,64) #input and output numbers
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10) #output labels

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x,dim=1)

net = Net()
print(net)

X = torch.rand((28,28))
output = net(X.view(-1,28*28)) #negative 1 specifies unknown shape
print(output)

optimizer = optim.Adam(net.parameters(),lr = 0.001)

EPOCHS = 4

for epoch in range(EPOCHS):
    for data in trainset:
        #data is a batch of featuresets and labels
        X,y =data
        net.zero_grad() #
        output =net(X.view(-1,28*28))
        loss = F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    print(loss)

for i in range(10):
    plt.imshow(X[i].view(28,28))
    print(torch.argmax(net(X[i].view(-1,28*28))[0]))

    plt.show()

