import torch
from segmentation_models_pytorch import DeepLabV2
from dataset import dataloader_train

NUM_CLASSES = 2

model = DeepLabV2(backbone='resnet101', classes=NUM_CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    # Addestramento per epoca
    model.train()
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
