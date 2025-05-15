import numpy as np
import utils.metrics as metrics
import torch
from torch.nn.utils import clip_grad_norm_

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    metrics_tracker = metrics.MLMetrics(objective='binary')

    for batch_idx, (inputs, aux_inputs, labels) in enumerate(train_loader):
        # inputs, aux_inputs, labels = inputs.float().to(device), aux_inputs.float().to(device), labels.float().to(device)
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        #inputs, labels = aux_inputs.float().to(device), labels.float().to(device)
        # 跳过所有标签相同的批次
        # if labels.sum() == 0 or labels.sum() == batch_size:
        #     continue

        optimizer.zero_grad()
        
        #outputs = model(inputs,aux_inputs)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 使用 with torch.no_grad() 避免不必要的计算图构建
        with torch.no_grad():
            probabilities = torch.sigmoid(outputs)
            labels_np = labels.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            metrics_tracker.update(labels_np, probabilities_np, [loss.item()])

        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return metrics_tracker

def validate(model, device, test_loader, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    all_losses = []

    with torch.no_grad():
        for batch_idx, (inputs, aux_inputs, labels) in enumerate(test_loader):
            #inputs, aux_inputs, labels = inputs.float().to(device), aux_inputs.float().to(device), labels.float().to(device)
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            #inputs, labels = aux_inputs.float().to(device), labels.float().to(device)
            #outputs = model(inputs, aux_inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probabilities = torch.sigmoid(outputs)

            all_labels.append(labels.to(device='cpu', dtype=torch.long).numpy())
            all_probs.append(probabilities.to(device='cpu').numpy())
            all_losses.append(loss.item())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_losses = np.array(all_losses)

    metrics_tracker = metrics.MLMetrics(objective='binary')
    metrics_tracker.update(all_labels, all_probs, [all_losses.mean()])

    return metrics_tracker, all_labels, all_probs