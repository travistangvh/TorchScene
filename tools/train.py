# modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
import copy
import logging
import os
import time
import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms
from meter import AverageMeter, ProgressMeter
from torch.utils.tensorboard import SummaryWriter

from miscellaneous import mkdir, collect_env_info
from model_zoo import initialize_model


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfgs: DictConfig):
    logger = logging.getLogger(cfgs.arch)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    writer = SummaryWriter(os.path.join(cfgs.tensorborad_log_dir, time.ctime().replace(' ', '_').replace(':', '-')))
    # create model
    logger.info("getting model '{}' from torch hub".format(cfgs.arch))
    model, input_size = initialize_model(
        model_name=cfgs.arch,
        num_classes=cfgs.num_classes,
        feature_extract=cfgs.feature_extract,
        use_pretrained=cfgs.pretrained,
    )
    logger.info("model: '{}' is successfully loaded".format(model.__class__.__name__))
    logger.info("model structure: {}".format(model))
    # Data augmentation and normalization for training
    # Just normalization for validation
    logger.info("Initializing Datasets and Dataloaders...")
    logger.info("loading data {} from {}".format(cfgs.dataset, cfgs.data_path))
    dataloaders_dict = load_data(
        input_size=input_size,
        batch_size=cfgs.batch_size,
        data_path=cfgs.data_path,
        num_workers=cfgs.workers
    )
    # Detect if we have a GPU available
    device = torch.device(cfgs.device if torch.cuda.is_available() else "cpu")

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    param_log_info = ''
    if cfgs.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                param_log_info += "\t{}".format(name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_log_info += "\t{}".format(name)
    logger.info("Params to learn:\n" + param_log_info)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=cfgs.lr, momentum=cfgs.momentum)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft = train_model(model, dataloaders_dict, device, criterion, optimizer_ft, logger,
                           print_freq=cfgs.print_freq, num_epochs=cfgs.epochs, is_inception=(cfgs.arch == "inception"),
                           tensorboard_plugin=writer
                           )
    mkdir(cfgs.weight_dir)
    torch.save(model_ft.state_dict(), os.path.join(cfgs.weight_dir, cfgs.arch) + '.ckpt')
    logger.info("model is saved at {}".format(os.path.abspath(os.path.join(cfgs.weight_dir, cfgs.arch) + '.ckpt')))


def load_data(input_size, data_path, batch_size, num_workers) -> dict[DataLoader, DataLoader]:
    """transform data and load data into dataloader. Images should be arranged in this way by default: ::

        root/my_dataset/train/dog/xxx.png
        root/my_dataset/train/dog/xxy.png
        root/my_dataset/train/dog/[...]/xxz.png

        root/my_dataset/val/cat/123.png
        root/my_dataset/val/cat/nsdf3.png
        root/my_dataset/val/cat/[...]/asd932_.png


    notice that the directory of your training data must be names as 'train', and
    the directory name of your validation data must be named as 'val', and they should
    under the same directory.

    Args:
        input_size (int): transformed image resolution, such as 224.
        data_path (string): eg. xx/my_dataset/
        batch_size (int): batch size
        num_workers (int): number of pytorch DataLoader worker subprocess
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in
                      ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ) for x in ['train', 'val']
    }
    return dataloaders_dict


def train_model(model, dataloaders, device, criterion, optimizer, logger, print_freq, num_epochs, tensorboard_plugin=None, is_inception=False):
    """a simple train and evaluate script modified from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html.

        Args:
            model (nn.Module): model to be trained.
            dataloaders (dict): should be a dict in the format of {'train': DataLoader, 'val': DataLoader}.
            device (Any): device.
            criterion (Any): loss function.
            optimizer (Any): optimizer.
            logger (Any): using logging.logger to print and log training information.
            print_freq (int): logging frequency.eg. 10 means logger will print information when 10 batches are trained or evaluated.
            num_epochs (int): training epochs
            tensorboard_plugin(Any): torch.utils.tensorboard.SummaryWriter
            is_inception (bool): please refer to https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
        """
    # Send the model to GPU
    model = model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # statistics
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.5f')
            top5 = AverageMeter('Acc@5', ':6.5f')
            progress = ProgressMeter(
                len(dataloaders[phase]),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            end = time.time()
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # measure accuracy and record loss
                acc1 = accuracy(outputs, labels, top_k=1)
                acc5 = accuracy(outputs, labels, top_k=5)
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1, inputs.size(0))
                top5.update(acc5, inputs.size(0))
                if phase == 'train':
                    tensorboard_plugin.add_scalar('loss/train', loss, i)
                    tensorboard_plugin.add_scalar('acc1/train', acc1, i)
                    tensorboard_plugin.add_scalar('acc5/train', acc5, i)
                else:
                    tensorboard_plugin.add_scalar('loss/val', loss, i)
                    tensorboard_plugin.add_scalar('acc1/val', acc1, i)
                    tensorboard_plugin.add_scalar('acc5/val', acc5, i)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % print_freq == 0:
                    logger.info(progress.display(i))
    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    main()
