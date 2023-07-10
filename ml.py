import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

path = "data/model.pth"
batch_size = 8 # seems like smaller batch sizes increase accuracy, so I moved from 64 to 8
# As long as it shows improvement, the model will be trained endlessly. 
# You can limit the maximum number of epochs here.
# If you want to get over it fast, set maxEpoch=1
# If you got time, set it to a value of 30 or so
# Make sure to delete the model.pth-file if you want to re-run the training
maxEpoch = 1

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print(f"Shape of x [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print( '****************************')
print(f"Using {device} device")
print( '****************************')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 28*28*9),
            nn.ReLU(),
            nn.Linear(28*28*9, 28*28*9),
            nn.ReLU(),
            nn.Linear(28*28*9, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def trainModel():
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    accuracy = 0
    currentEpoch = 0
    while True:
        currentEpoch += 1
        print(f"Epoch {currentEpoch}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        new_accuracy = test(test_dataloader, model, loss_fn)
        if (new_accuracy <= accuracy):
            break
        accuracy = new_accuracy
        if (accuracy > 99):
            break
        if (currentEpoch >= maxEpoch):
            break
    print(f"Best accuracy {accuracy}%")

model = NeuralNetwork().to(device)

if os.path.exists(path):
    model.load_state_dict(torch.load(path))
else:
    trainModel()
    torch.save(model.state_dict(), path)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
count = 0
correct = 0
incorrect = 0
with torch.no_grad():
    for i in range(0,len(test_data)):
        x, y = test_data[i][0], test_data[i][1]
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        if (predicted == actual):
            correct += 1
        else:
            incorrect += 1
        count += 1
        print(f'{count}. Predicted: "{predicted}", Actual: "{actual}"')
print(f'Total: {count}, Correct: {correct}, Incorrect: {incorrect}, Success: {(100 * correct / count):>0.2f}%')