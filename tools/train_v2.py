import logging
import time
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from models.vision_transformer import ViTB16
from utils.meter import AverageMeter, ProgressMeter


def train(train_loader, model, criterion, optimizer, epoch):
    logger = logging.getLogger("VITB16")
    logger.setLevel(logging.DEBUG)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, top_k=1)
        acc5 = accuracy(output, target, top_k=5)
        losses.update(loss, images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print(progress.display(i))


def main():
    traindir = "E://places365_standard/train/"
    valdir = "E://places365_standard/val/"
    resolution = [224, 224]
    batch_size = 1
    epoch = 7
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.Resize(
                        resolution,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=batch_size
    )

    model = ViTB16(num_classes=434)
    model.train()
    device = "cuda:0"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
    train(train_loader, model, criterion, optimizer, epoch)
    torch.save(model.state_dict(), "vit_base16.pt")


if __name__ == "__main__":
    main()
