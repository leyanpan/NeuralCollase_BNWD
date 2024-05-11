from tqdm import tqdm
import torch
import torch.nn.functional as F


# Train for one epoch
def train(model, criterion, device, num_classes, train_loader, optimizer, epoch, callback=None):
    model.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        if str(criterion) == 'CrossEntropyLoss()':
          loss = criterion(out, target)
        elif str(criterion) == 'MSELoss()':
          loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())

        loss.backward()
        optimizer.step()
        #if weight_reg and model_type == 'MLP':
        #  model.normalize_weight()
        accuracy = torch.mean((torch.argmax(out,dim=1)==target).float()).item()
        pbar.update(1)
        pbar.set_description(
            'Epoch: {} Batch Loss: {:.6f} Batch Accuracy: {:.6f}'.format(
                epoch,
                loss.item(),
                accuracy))
        if callback:
          callback(model, data, target)
    pbar.close()