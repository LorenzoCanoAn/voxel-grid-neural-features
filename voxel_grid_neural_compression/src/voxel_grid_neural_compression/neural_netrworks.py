import torch
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class VoxelGridCompressor(nn.Module):
    def __init__(self):
        super(VoxelGridCompressor, self).__init__()

        # Define 3D convolutional layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)

        # Define pooling and normalization layers
        self.max_pool = nn.MaxPool3d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.batchnorm = nn.BatchNorm3d(256)

        # Define fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        # Define comparator layers
        self.comp_fc1 = nn.Linear(16 * 2, 128)
        self.comp_fc2 = nn.Linear(128, 264)
        self.comp_fc3 = nn.Linear(264, 128)
        self.comp_fc4 = nn.Linear(128, 64)
        self.comp_fc5 = nn.Linear(64, 6)
        # Define dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def compress(self, x):
        x = self.max_pool(torch.relu(self.conv1(x)))
        x = self.max_pool(torch.relu(self.conv2(x)))
        x = self.max_pool(torch.relu(self.conv3(x)))
        x = self.batchnorm(self.max_pool(torch.relu(self.conv4(x))))
        x = self.avg_pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return x
    def compare(self, x, y):
        z = torch.cat((x,y),(-1))
        z = torch.relu(self.comp_fc1(z))
        z = self.dropout(z)
        z = torch.relu(self.comp_fc2(z))
        z = self.dropout(z)
        z = torch.relu(self.comp_fc3(z))
        z = self.dropout(z)
        z = torch.relu(self.comp_fc4(z))
        z = self.dropout(z)
        z = torch.relu(self.comp_fc5(z))
        return z
    def forward(self, x, y):
        # Convolutional layers
        x = self.compress(x)
        y = self.compress(y)
        # Fully connected layers
        z = self.compare(x,y)
        return z


class my_dataset(Dataset):
    def __init__(self, length = 100000):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        x = torch.randint(0,15,(1,100,100,30)).to(torch.float32).to('cuda')
        y = torch.randint(0,15,(1,100,100,30)).to(torch.float32).to('cuda')
        z = torch.randint(0,15,(6,)).to(torch.float32).to('cuda')
        return x,y,z

model = VoxelGridCompressor().to('cuda')
criterion = nn.MSELoss(6)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
dataset = my_dataset()
trainloader = DataLoader(dataset,batch_size=32)
for epoch in tqdm(range(5)):  # Change the number of epochs as needed
    running_loss = 0.0
    for vg1, vg2, labels in tqdm(trainloader):
        optimizer.zero_grad()
        outputs = model(vg1, vg2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()