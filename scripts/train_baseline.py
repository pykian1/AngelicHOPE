"""
Remark:

due to compute constraints (running on a macbook) this project will be utilizing the cifar10 dataset instead of a subclass of cifar100 
as seen in the paper

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
import os
import time
from hope_implementation.models import VGG8, get_device, load_frozen

DATA=os.path.expanduser("~/data/cifar10")
DOWNLOAD=True
EPOCHS=100
BATCH_SIZE=16 # paper specifices testing with batch_zize of 16 + LR of .05
LR=.05
SEED=0
ETA_MIN=5e-5
WEIGHT_DECAY=5e-4
MOMENTUM=.9
VAL_FRACTION=.2
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "vgg8_baseline.pt")
CKPT_PER = 10


MEAN=(.4914, .4822, .4465)
STD=(.2470, .2435, .2616)
test_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),  
    ]
)
training_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN,STD),

    ]
)


def build_param_groups(model, weight_decay):
# buckets pre-training parameters into decay and no_decay 
# excluding beta keeps the training regimen from adding noise to the effective bias
# that enters ReLU self-kernel

    decay, no_decay= [], []

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            decay.append(m.weight)
            if m.bias is not None:
                no_decay.append(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            decay.append(m.weight)
            no_decay.append(m.bias)
        
    ngrouped = sum(p.numel() for p in decay + no_decay)
    ntotal = sum(p.numel() for p in model.parameters())

    if ngrouped != ntotal:
        raise RuntimeError(
            f"parameter grouping mismatch: {ngrouped} vs {ntotal}, "
            f"missing {ntotal - ngrouped} params"
        )

    print(f"decay: {len(decay)} tensors | no_decay: {len(no_decay)} tensors")

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

def training_loop(dataloader, model, criterion, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    running = 0.0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss= criterion(model(X), y)
        loss.backward()
        optimizer.step()

        running+= loss.item()

        if batch%100==0:
            current= batch*dataloader.batch_size + len(X)
            print(f"    loss {loss.item():7.4f} [{current:>6d}/{size:>6d}]")
    return running / len(dataloader)

@torch.no_grad()
def testing_loop(dataloader, model, criterion, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    for X,y in dataloader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)

        test_loss += criterion(logits,y).item()
        correct+= (logits.argmax(1)== y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct/=size
    return test_loss, correct

def save(path, epoch, model, optimizer, scheduler):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        },
        path
    )





def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    _training_set = datasets.CIFAR10(root=DATA, train=True, transform=training_tf, download=DOWNLOAD)
    _val_set = datasets.CIFAR10(root=DATA, train=True, transform=test_tf, download=DOWNLOAD)
    testing_set = datasets.CIFAR10(root=DATA, train=False, transform=test_tf, download=DOWNLOAD)
    _n_total = len(_training_set)
    _n_val = int(_n_total*VAL_FRACTION)

    _gen =torch.Generator().manual_seed(SEED)
    perm = torch.randperm(_n_total, generator=_gen).tolist()
    val_idx, train_idx = perm[:_n_val], perm[_n_val:]
    train_set=Subset(_training_set, train_idx)
    val_set = Subset(_val_set, val_idx)

    loader_kwargs = dict(num_workers=4, pin_memory=False, persistent_workers=True) # pin_memory=False as this is being ran on MPS
    training_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(testing_set, batch_size=256, shuffle=False, **loader_kwargs)

    print(f"training: {len(train_set)}, validating: {len(val_set)}, testing: {len(testing_set)}")

    device = get_device()
    model = VGG8().to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(build_param_groups(model, WEIGHT_DECAY), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=ETA_MIN)

    best_acc = 0.0
    start = time.time()
    for i in range(EPOCHS):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n{'='*62}\nEpoch {i+1:>3d}/{EPOCHS} (lr {lr_now:.5f})\n{'=' * 20}")
        train_loss = training_loop(training_loader, model, criterion, optimizer, device)
        val_loss, val_acc = testing_loop(val_loader, model, criterion, device)
        
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        elapsed = (time.time() - start) / 60
        print(f" train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"acc: {100*val_acc:.2f}% | {elapsed:.1f} min")          
        if (i+1) % CKPT_PER == 0 or (i+1)==EPOCHS: 
            save(os.path.join(CKPT_DIR, f"vgg8_epoch{i+1}.pt"), i+1, model, optimizer, scheduler)
            print(f" resume checkpoint: vgg8_epoch{i+1}.pt")

    

    torch.save(model.state_dict(), CKPT_PATH)   

    test_loss, test_acc = testing_loop(test_loader, model, criterion, device)
    print(f"\n{'+' *62}")
    print(f"final test acc  {100 * test_acc:5.2f}% | (loss {test_loss:.4f})")
    print(f"best val acc    {100 * best_acc:5.2f}% ")
    print(f"frozen -> {CKPT_PATH}")

    reloaded = load_frozen(CKPT_PATH, device=device)
    model.eval()
    X, _ = next(iter(test_loader))
    X = X.to(device)
    with torch.no_grad():
        delta = (model(X) - reloaded(X)).abs().max().item()
    print(f"reload check: max |delta| = {delta:.2e}")
    assert delta < 1e-5, "checkpoint round_trip mismatch"


    #check how many BN gammas collapsed
    gammas = torch.cat([
        m.weight.detach().abs().flatten() for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    ])

    for thresh in (1e-2, 1e-3, 1e-4):
        frac = (gammas < thresh).float().mean().item()
        print(f" |gamma| < {thresh:.0e}: {100*frac:.1f}%")

    print("Done!")

if __name__ == "__main__":
    main()

