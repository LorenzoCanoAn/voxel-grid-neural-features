import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from voxel_grid_dataset.dataset import TwoVoxelGridsTwoTransformsDataset
from voxel_grid_neural_compression.neural_netrworks import VoxelGridCompressor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dataset = TwoVoxelGridsTwoTransformsDataset()
dataset.load_dataset()
dataloader = torch.utils.data.DataLoader(dataset)


criterion = nn.MSELoss(1)

# Training loop
trainloader = DataLoader(dataset, batch_size=16, shuffle=True)
writer = SummaryWriter("/home/lorenzo/.tensorboard")
counter = 0
for lr in [0.0005,0.0001,0.00005,0.00001,0.000005,0.000001]:
    model = VoxelGridCompressor().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in tqdm(range(32)):  # Change the number of epochs as needed
        for vg1, vg2, labels in tqdm(trainloader,leave=False):
            optimizer.zero_grad()
            vg1 = vg1.to("cuda")
            vg2 = vg2.to("cuda")
            labels = labels.to("cuda")
            outputs = model(vg1, vg2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar(f"loss_per_iter_lr{lr}", loss, counter)
            counter += 1
    torch.save(model.state_dict(), f"/home/lorenzo/models/a{lr}.torch")