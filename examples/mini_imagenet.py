import argparse
import os
import random
from collections import OrderedDict

from ResNet12 import Res12

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from matplotlib import pyplot as plt


class MiniImageModel(nn.Module):
    def __init__(self, args):
        super(MiniImageModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.conv3 = nn.Conv2d(50, 75, 5)
        self.fc1 = nn.Linear(75*7*7, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, args.dim)
        self.classifier = nn.Linear(args.dim, args.num_classes)

    def embed(self, x):
        # add dropout
        x = F.dropout(F.relu(self.conv1(x)), p=0.2)
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(F.relu(self.conv2(x)), p=0.2)
        x = F.max_pool2d(x, 2, 2)
        x = F.dropout(F.relu(self.conv3(x)), p=0.2)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 75*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.embed(x)
        return F.log_softmax(self.classifier(x), dim=-1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader),
                    100.0 * batch_idx / train_loader.num_batches(),
                    loss.item(),
                )
            )

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader),
            100.0 * correct / len(test_loader),
        )
    )

    score = correct / len(test_loader)
    return test_loss, score

class MiniImageNetBatchSampler:
    def __init__(self, split, batch_size, transform=None, rng=None):
        self.split = split
        self.batch_size = batch_size

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if rng is None:
            rng = random.Random()
        self.rng = rng

    def __iter__(self):
        self.rng.shuffle(self.split)
        
        batches = [
            self.split[i:i + self.batch_size]
            for i in range(0, len(self.split), self.batch_size)
        ]

        for batch in batches:
            batch_data = torch.stack([
                self.transform(Image.open(img_path).convert("L"))
                for img_path, _ in batch
            ])
            batch_target = torch.tensor([cls_idx for _, cls_idx in batch])
            yield batch_data, batch_target

    def num_batches(self):
        return np.ceil(len(self.split) / self.batch_size)

    def __len__(self):
        return len(self.split)

class MiniImageNet(Dataset):
    def __init__(self, root):
        self.root = root

        self.num_classes = len(os.listdir(root))
        self.class_name_map = OrderedDict(zip(os.listdir(root), np.arange(self.num_classes, dtype=int)))

        self.classes_samples = {}
        self.img_per_class = None
        for cls in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue

            cls_idx = self.class_name_map[cls]

            for img_name in os.listdir(cls_path):
                self.classes_samples.setdefault(cls_idx, []).append(os.path.join(cls_path, img_name))

            if self.img_per_class is None:
                self.img_per_class = len(self.classes_samples[cls_idx])
            else:
                assert self.img_per_class == len(self.classes_samples[cls_idx])

    def get_classe_name_map(self):
        return self.class_name_map

    def make_splits(self, id_classes, split_ratio=0.7, rng=None):
        if rng is None:
            rng = random.Random()

        splits = {
            "train": [],
            "id_test": [],
            "ood_test": [],
        }

        split_idx = int(self.img_per_class * split_ratio)
        cls_idx_to_label = {
            cls_idx: label for label, cls_idx in enumerate(id_classes)
        }

        for cls_idx in range(self.num_classes):
            cls_samples = [(img_path, cls_idx) for img_path in self.classes_samples[cls_idx]]
            rng.shuffle(cls_samples)

            if cls_idx in id_classes:
                cls_samples = [
                    (img_path, cls_idx_to_label[cls_idx])
                    for img_path, _ in cls_samples
                ]
                splits["train"] += cls_samples[:split_idx]
                splits["id_test"] += cls_samples[split_idx:]
            else:
                splits["ood_test"] += cls_samples

        for k in splits:
            rng.shuffle(splits[k])

        return splits

def precompute_transforms(splits):
    total = 0.0
    total_squared = 0.0
    pixel_count = 0
   
    for img_path, _ in splits["train"]:
        img = Image.open(img_path).convert("L")
        img_tensor = transforms.ToTensor()(img)

        total += img_tensor.sum().item()
        total_squared += img_tensor.pow(2).sum().item()
        pixel_count += img_tensor.numel()

    mean = total / pixel_count
    variance = max(total_squared / pixel_count - mean ** 2, 0.0)
    std = variance ** 0.5

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    return transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini ImageNet")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimension of embeddings"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, help="Number of classes"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for saving the model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    CWD = os.getcwd()
    
    mini_imagenet_path = os.path.join(CWD, "mini-imagenet")
    assert os.path.isdir(mini_imagenet_path)

    mini_imagenet_path_sets = {}
    for key in ("train", "test", "val"):
        mini_imagenet_path_sets[key] = os.path.join(mini_imagenet_path, key)
        assert os.path.isdir(mini_imagenet_path_sets[key])
    
        # train contains 64 classes
        # test contains 20 classes
        # val contains 16 classes
        print(f"{key} contains {len(os.listdir(mini_imagenet_path_sets[key]))} classes")

    ds = MiniImageNet(mini_imagenet_path_sets["train"])
    classes = list(ds.get_classe_name_map().values())

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    id_classes = rng.sample(classes, args.num_classes)
    print(f"In domain classes: {id_classes}")

    splits = ds.make_splits(id_classes, rng=rng)
    for k, v in splits.items():
        print(f"{k} contains {len(v)} samples")

    transform = precompute_transforms(splits)
    train_sampler = MiniImageNetBatchSampler(splits["train"], args.batch_size, transform=transform, rng=rng)
    test_sampler = MiniImageNetBatchSampler(splits["id_test"], args.test_batch_size, transform=transform, rng=rng)

    # model = MiniImageModel(args).to(device)
    model = Res12().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # for batch_idx, (data, target) in enumerate(train_sampler):
    #     print(data.shape)
    #     print(target)
    #     break

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_sampler, optimizer, epoch)
        test(args, model, device, test_sampler)