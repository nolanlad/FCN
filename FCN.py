# emnist with torch


import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from emnist import *
import string

lets = string.ascii_lowercase

n_classes = len(lets)

letsid = lets[:n_classes]

def im2tensor(x):
    x = torch.tensor([[x]])
    x = x/torch.max(x)
    return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 120, kernel_size=5)
        self.drop = nn.Dropout(0.2)
        self.norm1 = nn.BatchNorm2d(120)
        self.norm12 = nn.BatchNorm2d(60)
        self.norm2 = nn.BatchNorm2d(n_classes)
        self.conv12 = nn.Conv2d(120, 60, kernel_size=5)
        self.conv2 = nn.Conv2d(60, n_classes, kernel_size=5)
        self.softmax = nn.Softmax(dim =1)
        self.maxpool = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = self.drop(x)
        x = self.norm12(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.maxpool(x)
        x = x.view(-1,n_classes)
        x = self.softmax(x)
        return x
    
    def get_map(self,x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.norm2(x)
        return x

net = Net()
#criterion = nn.L1Loss()
criterion = nn.MSELoss()
#criterion = DiceBCELoss()
#criterion = nn.CTCLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

hotvecs = []
data = []
for k in letsid:
    for i in range(len(dic[k])):
        vec = np.zeros(len(letsid))
        j = lets.find(k)
        vec[j] = 1.
        hotvecs.append(vec)
        data.append([dic[k][i]])

arr = np.arange(len(data),dtype=int)
np.random.shuffle(arr)
data = np.array(data)
hotvecs = np.array(hotvecs)

data = data[arr]
hotvecs = hotvecs[arr]

data = torch.tensor(data)
data = data/torch.tensor(255)

hotvecs = torch.tensor(hotvecs,dtype=torch.float32)


trainlen = int(0.8*data.shape[0])

# batchsize=8

# for epoch in range(2):
#     for i in range(0,trainlen,batchsize):
#         for k in letsid:
            
            
#             optimizer.zero_grad()

#             ls = np.random.choice(arr,batchsize)
#             #dat = data[i:(i+batchsize)]
#             #label = hotvecs[i:(i+batchsize)]
#             dat = data[ls]
#             label = hotvecs[ls]
#             dat = dat*0.9
#             dat = dat + 0.1*torch.rand(dat.shape)

#             outputs = net(dat)
#             loss = criterion(outputs, label)
#             loss.backward()
#             optimizer.step()
#     print(epoch,loss)

# batchsize = 16
# np.random.shuffle(arr)

# for epoch in range(2):
#     for i in range(0,trainlen,batchsize):
#         for k in letsid:

#             optimizer.zero_grad()

#             ls = np.random.choice(arr,batchsize)
#             dat = data[ls]
#             label = hotvecs[ls]
#             dat = dat*0.9
#             dat = dat + 0.1*torch.rand(dat.shape)
#             #dat = data[i:(i+batchsize)]
#             #label = hotvecs[i:(i+batchsize)]

#             outputs = net(dat)
#             loss = criterion(outputs, label)
#             loss.backward()
#             optimizer.step()
#         print(epoch,loss)

batchsize = 16
np.random.shuffle(arr)

for epoch in range(1):
    np.random.shuffle(arr)
    for i in range(0,trainlen,batchsize):

        optimizer.zero_grad()

        # ls = np.random.choice(arr,batchsize)
        # dat = data[ls]
        # label = hotvecs[ls]
        dat = data[i:(i+batchsize)]
        label = hotvecs[i:(i+batchsize)]
        dat = dat*0.9
        dat = dat + 0.1*torch.rand(dat.shape)
        #dat = data[i:(i+batchsize)]
        #label = hotvecs[i:(i+batchsize)]

        outputs = net(dat)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        print(epoch,loss)

 
def test():
    test = data[trainlen:]
    testlabs = hotvecs[trainlen:]
    outputs = net(test)
    outputs2 = torch.argmax(outputs,axis=1)
    outputs3 = torch.argmax(testlabs,axis=1)
    j = outputs2 == outputs3
    right = torch.sum(j)
    print(right/(test.shape[0]))


test()
