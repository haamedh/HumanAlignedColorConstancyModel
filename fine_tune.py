import torch
import model
import datasetImage as dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from util import Logger
import loss
import transforms as T


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    step_size = 200
    gamma = 0.8
    n_classes = 3
    learning_rate = 1e-3
    logs_per_epoch = 2
    model_name = "lab100"

    trans = T.Compose3([
        T.ToTensor3(),
    ])

    indoor_dir = r"dataset/indoor448/Control1/"
    outdoor_dir = r"dataset/outdoor448/Control/"
    train_dataset = dataset.ColorConstancyDataset(indoor_dir, outdoor_dir, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    net = model.UNetWithResnet50Encoder(n_classes=n_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = loss.PBCLoss(reduction='none')

    weights = torch.load('pretrained_weight/model_weights.pth', map_location='cuda:0')
    net.load_state_dict(weights['state_dict'])
    optimizer.load_state_dict(weights['optimizer'])

    net = net.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # freeze early encoder blocks
    for i, block in enumerate(net.down_blocks):
        if i < 6:
            for param in block.parameters():
                param.requires_grad = False

    logger = Logger()
    logger.log(f"Model: {model_name}")
    logger.log(f"Batch Size: {batch_size}")
    logger.log(f"Step Size: {step_size}")
    logger.log(f"Gamma: {gamma}")
    log_interval = len(train_loader) // logs_per_epoch

    for epoch in range(1, 4001):
        net.train()
        logger.log(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        running_loss = 0.0
        epoch_loss = 0.0
        n_epoch_samples = 0
        n_step_samples = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for batch_idx, data in pbar:
            indexes, filenames, inputs, labels, seg = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            segs = seg.to(device)

            outputs = net(inputs)

            batch_loss = criterion(outputs, labels)
            batch_loss = batch_loss.unsqueeze(1)
            weight_mask = torch.where(segs > 0, 1, 1)
            weighted_loss = batch_loss * weight_mask
            batch_loss = weighted_loss.sum()

            running_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            n_epoch_samples += len(inputs)
            n_step_samples += len(inputs)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix(Loss=batch_loss.item())

            if batch_idx % log_interval == 0:
                logger.log(f"Epoch: {epoch}, Iteration: {batch_idx}, Loss: {running_loss / n_step_samples:.4f}")
                running_loss = 0.0
                n_step_samples = 0

        logger.log(f"Epoch: {epoch}, Loss: {epoch_loss / n_epoch_samples:.4f}")
        scheduler.step()
        logger.log_image_lab(inputs, labels, outputs, epoch, batch_idx, filenames[0])

        if epoch % 200 == 0:
            logger.save_checkpoint(net, optimizer, scheduler, epoch)

        print(batch_idx)
