import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


data_path = "../mnist_data"
if os.getenv("GLOBALSCRATCH") is not None:
    data_path = os.path.join(os.getenv("GLOBALSCRATCH"), "mnist_data")

fix_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.urls = [
    fix_mirror + '/' + md5
    for _, md5 in (u.rsplit('/', 1) for u in datasets.MNIST.urls)
]


class AE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_dim = 32*7*7
        self._to_latent = nn.Linear(conv_dim, args.dim)
        self._from_latent = nn.Linear(args.dim, conv_dim)

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = self._encoder(x)
        z = torch.flatten(z, 1)
        return self._to_latent(z)

    def decode(self, z):
        z = self._from_latent(z)
        z = z.view(-1, 32, 7, 7)
        return self._decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_dim = 32*7*7
        self._to_mu       = nn.Linear(conv_dim, args.dim)
        self._to_logvar   = nn.Linear(conv_dim, args.dim) 
        self._from_latent = nn.Linear(args.dim, conv_dim)

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = self._encoder(x)
        z = torch.flatten(z, 1)
        return self._to_mu(z), self._to_logvar(z)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        z = self._from_latent(z)
        z = z.view(-1, 32, 7, 7)
        return self._decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z               = self.reparam(mu, logvar)
        recon           = self.decode(z)

        return recon, mu, logvar


def VAE_loss(data, recon, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, data, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / data.size(0)


def train_VAE(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        loss = VAE_loss(data, *model(data))
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_VAE(args, model, device, test_loader):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += VAE_loss(data, *output)
            
    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}\n".format(
            test_loss
        )
    )

    return test_loss


def train_AE(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_AE(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    total_pixels = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(
                output, data, reduction="sum"
            ).item()
            total_pixels += data.numel()
            
    test_loss /= total_pixels

    print(
        "\nTest set: Average loss: {:.4f}\n".format(
            test_loss
        )
    )

    return test_loss


def run(train_loader, test_loader, args, device, target_loss=None):
    model = VAE(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    min_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_VAE(args, model, device, train_loader, optimizer, epoch)
        loss = test_VAE(args, model, device, test_loader)
        decreased = loss < min_loss

        if decreased:
            min_loss = loss
            best_epoch = epoch

        if args.save_model and decreased:
            torch.save(model.state_dict(), f"checkpoints/{args.run_id}/mnist_vae_{epoch}_{args.dim}d.pt")

        if target_loss is not None and loss < target_loss:
            print(f"Target loss of {target_loss:.6f} reached")
            break

    return best_epoch, min_loss


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST VAE")
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
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for saving the model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), 
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(), 
            ]),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    if args.save_model:             
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"checkpoints/{args.run_id}", exist_ok=True)

    best_epoch, _ = run(train_loader, test_loader, args, device)
    appended_args = {**vars(args), "best_epoch": best_epoch}

    if args.save_model:
        with open(f"checkpoints/{args.run_id}/mnist_args_{args.dim}d.json", "w") as f:
            json.dump(appended_args, f)

if __name__ == "__main__":
    main()
