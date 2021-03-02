import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

import time

from models import ConvNet, MLP
from data_prepare import svhn_transform_train, svhn_transform_test
from copy import deepcopy

def get_num_classes(task):
    if task in ["uu-ent-down", "dd-mis-down", "uu-ent-up1", "uu-ent-up2", "dd-mis-up"]:
        num_classes = 2
    elif task == "lf-mis-up":
        num_classes = 1
    elif task == "lf-mis-down":
        num_classes = 4
    elif task in ["hs-mis-up", "hs-mis-down"]: 
        num_classes = 10
    return num_classes

def get_model(model, num_classes, linear_base=None, device="cuda"):
    if model.startswith('convnet'):
        net = ConvNet(num_classes=num_classes)
        if len(model.split("-")) >= 2:
            hidden_layers = [int(num_h) for num_h in model.split("-")[1:]]
            net.fc1 = MLP(net.fc1.in_features, net.fc1.out_features, hidden_layers)
    elif model.startswith("resnet18"):
        net = torchvision.models.resnet18(num_classes=num_classes)
        if len(model.split("-")) >= 2:
            hidden_layers = [int(num_h) for num_h in model.split("-")[1:]]
            net.fc = MLP(net.fc.in_features, net.fc.out_features, hidden_layers)
    elif model == "linear":
        net = nn.Linear(in_features=linear_base, out_features=num_classes)
    elif model.startswith("MLP"):
        hidden_layers = [int(num_h) for num_h in model.split("-")[1:]]
        net = MLP(linear_base, num_classes, hidden_layers)
    return net.to(device)

def get_dataset(task, para_to_vary_model, seed=0, data_root="~/data"):
    transform_train = svhn_transform_train
    transform_test = svhn_transform_test
    trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform_test)

    #modify labels for different tasks
    if task == "dd-mis-up":
        def modify_label(dataset):
            selected_idx_0 = np.concatenate([np.nonzero(dataset.labels == i)[0] for i in [1, 0, 7, 4, 9]])
            selected_idx_1 = np.concatenate([np.nonzero(dataset.labels == i)[0] for i in [6, 2, 3, 5, 8]])
            dataset.labels[selected_idx_0] = 0
            dataset.labels[selected_idx_1] = 1
            return dataset
        trainset = modify_label(trainset)
        testset = modify_label(testset)
    elif task in ["uu-ent-down", "uu-ent-up1", "uu-ent-up2"]:
        np.random.seed(0)
        def select_data(dataset):
            selected_idx = [np.nonzero(dataset.labels == i)[0] for i in [1, 7, 6]]
            class_size = min(len(a) for a in selected_idx)
            selected_idx = np.concatenate([a[np.random.choice(len(a), class_size, replace=False)] for a in selected_idx])
            dataset.labels = dataset.labels[selected_idx]
            dataset.data = dataset.data[selected_idx]
            return dataset
        trainset = select_data(trainset)
        testset = select_data(testset)
        if task == "uu-ent-down":
            trainset.labels[trainset.labels == 1] = 1
            trainset.labels[trainset.labels == 6] = 0
            trainset.labels[trainset.labels == 7] = 1
            testset.labels[testset.labels == 1] = 1
            testset.labels[testset.labels == 6] = 0
            testset.labels[testset.labels == 7] = 1
    elif task == "dd-mis-down":
        def select_data(dataset):
            selected_idx = [np.nonzero(dataset.labels == i)[0] for i in [1, 7]]
            for i in range(len(selected_idx)):
                dataset.labels[selected_idx[i]] = i
            selected_idx = np.concatenate(selected_idx)
            dataset.labels = dataset.labels[selected_idx]
            dataset.data = dataset.data[selected_idx]
            return dataset
        trainset = select_data(trainset)
        testset = select_data(testset)
    elif task == "lf-mis-up":
        np.random.seed(0)
        def select_data(dataset):
            selected_idx = [np.nonzero(dataset.labels == i)[0] for i in [1, 3, 7, 9]]
            class_size = min(len(a) for a in selected_idx)
            selected_idx = np.concatenate([a[np.random.choice(len(a), class_size, replace=False)] for a in selected_idx])
            dataset.labels = dataset.labels[selected_idx]
            dataset.data = dataset.data[selected_idx]
            return dataset
        trainset = select_data(trainset)
        testset = select_data(testset)
    elif task == "lf-mis-down":
        np.random.seed(0)
        def select_data(dataset):
            selected_idx = [np.nonzero(dataset.labels == i)[0] for i in [1, 3, 7, 9]]
            for i in range(len(selected_idx)):
                dataset.labels[selected_idx[i]] = i
            class_size = min(len(a) for a in selected_idx)
            selected_idx = np.concatenate([a[np.random.choice(len(a), class_size, replace=False)] for a in selected_idx])
            dataset.labels = dataset.labels[selected_idx]
            dataset.data = dataset.data[selected_idx]
            return dataset
        trainset = select_data(trainset)
        testset = select_data(testset)

    #vary training sets
    if para_to_vary_model is not None:
        if task in ["uu-ent-up1", "uu-ent-up2"]:
            FE_rank = torch.load("aul_for_uu-ent.pth")
            selected_idx = FE_rank[:int(0.5 * para_to_vary_model * len(FE_rank))]
            trainset.labels[selected_idx] = (8 - trainset.labels[selected_idx]).astype(np.int)
            if task in "uu-ent-up1":
                trainset.labels[trainset.labels == 1] = 1
                trainset.labels[trainset.labels == 6] = 0
                trainset.labels[trainset.labels == 7] = 0
                testset.labels[testset.labels == 1] = 1
                testset.labels[testset.labels == 6] = 0
                testset.labels[testset.labels == 7] = 0
            else:
                trainset.labels[trainset.labels == 1] = 0
                trainset.labels[trainset.labels == 6] = 0
                trainset.labels[trainset.labels == 7] = 1
                testset.labels[testset.labels == 1] = 0
                testset.labels[testset.labels == 6] = 0
                testset.labels[testset.labels == 7] = 1
        elif task in ["dd-mis-up", "hs-mis-up"]:
            np.random.seed(seed)
            rand_perm = np.random.permutation(len(trainset))
            sampled_idx = rand_perm[:int(para_to_vary_model * len(trainset))]
            trainset.labels = trainset.labels[sampled_idx]
            trainset.data = trainset.data[sampled_idx]
        elif task == "lf-mis-up":
            np.random.seed(seed)
            def subsample(dataset, para_to_vary_model):
                rand_perm = np.random.permutation(len(dataset))
                clean_idx = rand_perm[: int(0.1 * len(rand_perm))]
                dirty_idx = rand_perm[int(0.1 * len(rand_perm)) : int((para_to_vary_model) * len(rand_perm))]
                dataset.data = dataset.data[np.concatenate([clean_idx, dirty_idx])]
                clean_label = dataset.labels[clean_idx]
                dirty_label = dataset.labels[dirty_idx]
                dirty_idx_flip = np.concatenate([np.nonzero(dirty_label == 1)[0], np.nonzero(dirty_label == 3)[0]])
                dirty_idx_flip = dirty_idx_flip[np.random.choice( len(dirty_idx_flip), int(len(dirty_idx_flip) * 0.5), replace=False)]
                dirty_label[dirty_idx_flip] = 4 - dirty_label[dirty_idx_flip]

                dirty_idx_flip = np.concatenate([np.nonzero(dirty_label == 7)[0], np.nonzero(dirty_label == 9)[0]])
                dirty_idx_flip = dirty_idx_flip[np.random.choice( len(dirty_idx_flip), int(len(dirty_idx_flip) * 0.5), replace=False)]
                dirty_label[dirty_idx_flip] = 16 - dirty_label[dirty_idx_flip]
                dataset.labels[np.concatenate([clean_idx, dirty_idx])] = np.concatenate([clean_label, dirty_label])
                dataset.labels = dataset.labels[np.concatenate([clean_idx, dirty_idx])]
                return dataset
            trainset = subsample(trainset, para_to_vary_model)
    return trainset, testset


def split_test_val(testloader):
    testset = testloader.dataset
    batch_size = testloader.batch_size
    img_inds = np.arange(len(testset))
    np.random.seed(0)
    np.random.shuffle(img_inds)
    test_inds = img_inds[:int(0.5 * len(img_inds))]
    val_inds = img_inds[int(0.5 * len(img_inds)):]
    testloader = data_utils.DataLoader(testset, num_workers=2, batch_size=batch_size, sampler=data_utils.SubsetRandomSampler(test_inds))
    valloader = data_utils.DataLoader(testset, num_workers=2, batch_size=batch_size, sampler=data_utils.SubsetRandomSampler(val_inds))
    np.random.seed(int(time.time()))
    return testloader, valloader


def get_criterion(task):
    if task == "lf-mis-up":
        criterion = torch.nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss() 
    return criterion


def get_repre(outputs, repre, device):
    if repre == "preds":
        _, preds = outputs.max(1)
        new_outputs = torch.zeros(outputs.size()).to(device)
        new_outputs[torch.arange(outputs.size()[0]).long(), preds] = 1
    elif repre in ["logits", "feat"]:
        new_outputs = outputs
    else:
        print("invalid representation")
        exit(0)
    return new_outputs


def get_output_from_model(inputs, net, repre, model, device):
    x = get_repre(torch.flatten(net(inputs), 1), repre, device)
    return x


def construct_tensor_dataset(dataloader, nets, repres, models, batch_size, device="cuda"):
    num_data = len(dataloader.dataset)
    output_tensor = None
    label_tensor = None
    with torch.no_grad():
        for net, model in zip(nets, models):
            net.eval()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.cat([get_output_from_model(inputs, net, repre, model, device) for net, repre, model in zip(nets, repres, models)], dim=1)
            if output_tensor is None:
                output_tensor = torch.zeros([num_data, outputs.size()[-1]])
                label_tensor = torch.zeros([num_data])
            output_tensor[batch_size * batch_idx : batch_size * (batch_idx + 1)] = outputs
            label_tensor[batch_size * batch_idx : batch_size * (batch_idx + 1)] = targets
    label_tensor = label_tensor.long()

    tensor_dataset = data_utils.TensorDataset(output_tensor, label_tensor)
    return tensor_dataset


def get_upstream_model(upstream_setting, linear_base=None, save_dir="./checkpoint/", device="cuda"):
    (upstream_task, upstream_para_to_vary_model, upstream_model, upstream_seed, repre, model_specify) = tuple(upstream_setting)
    upstream_num_classes = get_num_classes(upstream_task)
    upstream_net = get_model(upstream_model, upstream_num_classes, device=device, linear_base=linear_base)
    checkpoint = torch.load(f'{save_dir}/checkpoint/svhn_{upstream_model}_task_{upstream_task}_upstream_setting_None_para_to_vary_model_{upstream_para_to_vary_model}_seed_{upstream_seed}.pth')
    
    if model_specify == "last":
        upstream_net.load_state_dict(checkpoint['net'], strict=False)
    elif model_specify == "best":
        upstream_net.load_state_dict(checkpoint['best_state']['net'], strict=False)
    else:
        print("invalid upstream model specification")
        exit(0)
        
    if repre == "feat":
        if upstream_model.startswith("convnet"):
            upstream_net.fc1 = nn.Identity()
        else: 
            upstream_net.fc = nn.Identity()
    return upstream_net
    
        
def get_upstream_preprocessing(trainloader, testloader, upstream_type, save_dir, device):
    batch_size = trainloader.batch_size
    upstream_setting = upstream_type.split("_")
    num_upstream_model = int(len(upstream_setting) / 6)
    upstream_nets = []
    repres = []
    models = []
    for i in range(num_upstream_model):
        linear_base = next(iter(testloader))[0].view(batch_size, -1).size()[1]
        upstream_nets.append(get_upstream_model(upstream_setting[i * 6 : (i + 1) * 6], linear_base=linear_base, save_dir=save_dir, device=device))
        repres.append(upstream_setting[i * 6 + 4])
        models.append(upstream_setting[i * 6 + 2])

    new_trainset = construct_tensor_dataset(trainloader, upstream_nets, repres, models, batch_size, device=device)
    new_testset = construct_tensor_dataset(testloader, upstream_nets, repres, models, batch_size, device=device)
    trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(new_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def train_epoch(optimizer, criterion, trainloader, net, logger, regression, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if regression:
            targets = targets.float()
            loss = criterion(outputs[:, 0], targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        else:
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
    if regression:
        logger.info('==>>> train loss: {:.6f}'.format(train_loss/(batch_idx+1)))
    else:
        logger.info('==>>> train loss: {:.6f}, accuracy: {:.4f}'.format(train_loss/(batch_idx+1), 100.*correct/total))
        
        
def test_epoch(criterion, testloader, net, logger, regression, device):
    net.eval()
    batch_size = testloader.batch_size
    test_loss = 0
    correct = 0
    total = 0
    if isinstance(testloader.sampler, data_utils.SubsetRandomSampler):
        test_size = len(testloader.sampler.indices)
    else:
        test_size = len(testloader.dataset)
    
    loss_detail = torch.zeros([test_size])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if regression:
                targets = targets.float()
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                loss_detail[batch_idx * batch_size : (batch_idx + 1) * batch_size] = torch.abs(outputs[:, 0] - targets) ** 2
            else:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                loss_detail[batch_idx * batch_size : (batch_idx + 1) * batch_size] = 1 - predicted.eq(targets).float()
                correct += predicted.eq(targets).sum().item()
                
        if regression:
            logger.info('==>>> test loss: {:.6f}'.format(loss_detail.mean()))
            return loss_detail.mean(), loss_detail
        else:
            logger.info('==>>> test loss: {:.6f}, accuracy: {:.4f}, test zero-one loss: {:.4f}'.format(test_loss/(batch_idx+1), 100.*correct/total, loss_detail.mean()))
            return loss_detail.mean(), loss_detail


def get_dist(feat_set_1, feat_set_2, dist_type="cos"):
    if dist_type == "l2":
        dist = euclidean_distances(feat_set_1, feat_set_2)
    elif dist_type == "cos":
        dist = cosine_distances(feat_set_1, feat_set_2)
    return dist

def test_by_nn(testloader, net, model, logger, device):
    net.eval()
    batch_size = testloader.batch_size
    knn_net = deepcopy(net)
    feat_detail = None
    testset_size = len(testloader.dataset)
    target_detail = torch.zeros([testset_size])
    if model.startswith("convnet"):
        knn_net.fc1 = nn.Identity()
    else:
        knn_net.fc = nn.Identity()
    knn_net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = knn_net(inputs)
            if feat_detail is None:
                feat_detail = torch.zeros([testset_size, outputs.size()[1]])
            feat_detail[batch_idx * batch_size : (batch_idx + 1) * batch_size] = outputs
            target_detail[batch_idx * batch_size : (batch_idx + 1) * batch_size] = targets
        np.random.seed(0)
        randperm = np.random.permutation(len(testloader.dataset))
        val_split, test_split = randperm[:int(0.5 * testset_size)], randperm[int(0.5 * testset_size):]
        np.random.seed(int(time.time()))
        del knn_net
        val_feat, val_target = feat_detail[val_split].detach().cpu().numpy(), target_detail[val_split]
        test_feat, test_target = feat_detail[test_split].detach().cpu().numpy(), target_detail[test_split]
        test_to_val_dist = get_dist(test_feat, val_feat)
        test_preds = val_target[np.argsort(test_to_val_dist)[:, 0]]
        correct = test_preds.eq(test_target).sum().item()
        total = len(test_split)
        logger.info('==>>>knn zero-one loss: {:.4f}'.format(1 - float(correct)/total))
        return 1 - float(correct)/total