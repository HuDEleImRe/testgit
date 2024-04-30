import json
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch import utils, nn
from network import Hashnet, SupCon_CAResnet, Hash_resnet
from util import set_optimizer, adjust_learning_rate, warmup_learning_rate, save_model, config_dataset, get_data
from util import CELoss, AverageMeter
from cal_map import calculate_map, compress, calculate_top_map, pr_curve, CalcTopMapWithPR, compute_result
# from asymmetric_loss import AsymmetricLossSingleLabel
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        # "p": 1,
        "p": 2,
        "alpha":0.1,
        "beita":0.1,
        "gamma":0.5,
        "bit": 48,
        "resize_size": 32,
        "batch_size":64,
        "dataset": "cifar10-1",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "save_path": "save/DRNSH",
        "num_epoch":15,
        "test_map":15,
        "topK":-1
    }

    config = config_dataset(config)
    return config

def set_loader():
    config = get_config()
    train_loader, val_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    return train_loader, val_loader, dataset_loader, num_train, num_test, num_dataset

def set_model(ckpt, config):
    model = SupCon_CAResnet()
    # criterion = torch.nn.CrossEntropyLoss()
    bit = config["bit"]
    criterion = Loss(config, bit)
    hashcanet = Hash_resnet(encode_length=bit, num_classes=10)
    ckpt = torch.load(ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        hashcanet = hashcanet.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, hashcanet, criterion

class BCEQuantization(nn.Module):
    def __init__(self, std):
        super(BCEQuantization, self).__init__()
        self.BCE = nn.BCELoss()
        self.std=std
    def normal_dist(self, x, mean, std):
        prob = torch.exp(-0.5*((x-mean)/std)**2)
        return prob
    def forward(self, x):
        x_a = self.normal_dist(x, mean=1.0, std=self.std)
        x_b = self.normal_dist(x, mean=-1.0, std=self.std)
        y = (x.sign().detach() + 1.0) / 2.0
        l_a = self.BCE(x_a, y)
        l_b = self.BCE(x_b, 1-y)

        return (l_a + l_b)

class Loss(torch.nn.Module):
    def __init__(self, config, bit):
        super(Loss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.criterion = BCEQuantization(0.5)
    def hash_loss(self, hash_out, target):
        theta = torch.einsum('ij,jk->ik', hash_out, hash_out.t()) / 2
        # one_hot = torch.nn.functional.one_hot(target, self.num_classes)
        # one_hot = one_hot.float()
        one_hot = target
        Sim = (torch.einsum('ij,jk->ik', one_hot, one_hot.t()) > 0).float()


        pair_loss = (torch.log(1 + torch.exp(theta)) - Sim * theta)

        mask_positive = Sim > 0
        mask_negative = Sim <= 0
        S1 = mask_positive.float().sum() - hash_out.shape[0]
        S0 = mask_negative.float().sum()
        if S0 == 0:
            S0 = 1
        if S1 == 0:
            S1 = 1
        S = S0 + S1
        pair_loss[mask_positive] = pair_loss[mask_positive] * (S / S1)
        pair_loss[mask_negative] = pair_loss[mask_negative] * (S / S0)

        diag_matrix = torch.tensor(np.diag(torch.diag(pair_loss.detach()).cpu())).cuda()
        pair_loss = pair_loss - diag_matrix
        count = (hash_out.shape[0] * (hash_out.shape[0] - 1) / 2)

        return pair_loss.sum() / 2 / count

    def forward(self, u, Y, y, ind, config):

        # x =x.clamp(min=-1, max=1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()
        b = u.sign()
        s = (y @ self.Y.t() > 0).float()

        inner_product = u @ self.U.t() * 0.5
        # likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        # likelihood_loss = likelihood_loss.mean()
        likelihood_loss = self.hash_loss(u,y)

        Classification_criteria = nn.CrossEntropyLoss()
        # Classification_criteria = AsymmetricLossSingleLabel()
        classification_loss  = Classification_criteria(Y, y)

        if config["p"] == 1:
            quantization_loss = config["alpha"] * u.mean(dim=1).abs().mean()
        else:
            quantization_loss = config["alpha"] * u.mean(dim=1).pow(2).mean()

        balancing_loss = config["gamma"] * (u - b).pow(2).mean()
        # balancing_loss = self.criterion(u) * config["gamma"]
        return  likelihood_loss + config["beita"]*classification_loss + quantization_loss + balancing_loss

def train(train_loader, model, hashcanet, criterion, optimizer, epoch, config):
    """one epoch training"""
    model.eval()
    hashcanet.train()
    train_loss = 0
    for idx, (images, labels, ind) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(epoch, idx, len(train_loader), optimizer, task='hashnet')

        # compute loss
        with torch.no_grad():
            features = model.encoder(images) #(256, 2048)
        u, out = hashcanet(features.detach())

        loss = criterion(u, out, labels.float(), ind, config)
        train_loss += loss

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss

def main():

    start_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    ckpt = "cifir_new0107.pth"
    # build data loader
    train_loader, val_loader, dataset_loader, num_train,  num_test, num_dataset = set_loader()

    config = get_config()
    config['num_train'] = num_train
    config['n_class'] = 10
    print(config)

    # build models and criterion
    model, hashnet, criterion = set_model(ckpt, config)
    model.cuda()

    # build optimizer
    optimizer = set_optimizer(hashnet,config)

    # training routine
    print("<---开始训练--->")
    best_map = 0
    for epoch in range(1, config["num_epoch"] + 1):
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        start_train = time.time()
        loss = train(train_loader, model, hashnet, criterion,
                          optimizer, epoch, config)
        end_train = time.time()
        print('epoch {}/{}, loss {:.4f}, train time:{:.4f}'.format(epoch, config['num_epoch'], loss, end_train-start_train))
        if epoch % config["test_map"] == 0:

            device = config["device"]
            Start = time.time()
            tst_binary, tst_label = compute_result(val_loader, model, hashnet, device=device)

            # print("calculating dataset binary code.......")
            trn_binary, trn_label = compute_result(dataset_loader, model, hashnet, device=device)

            # mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
            mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                         trn_binary.numpy(), trn_label.numpy(), config["topK"])
            End = time.time()

            if mAP > best_map:
                best_map = mAP
            print("mAP:{}, computing time:{:.4f}, best mAP:{}".format(mAP, End - Start, best_map))
    print("<---训练结束--->")
    # print("<---PR_cruve--->")
    # index_range = num_dataset // 100
    # index = [i * 100 - 1 for i in range(1, index_range + 1)]
    # max_index = max(index)
    # overflow = num_dataset - index_range * 100
    # index = index + [max_index + i for i in range(1, overflow + 1)]
    # c_prec = cum_prec[index]
    # c_recall = cum_recall[index]
    #
    # pr_data = {
    #     "index": index,
    #     "P": c_prec.tolist(),
    #     "R": c_recall.tolist()
    # }
    # pr_curve_path = "chart/cifar/DRNSH-12.json"
    # os.makedirs(os.path.dirname(pr_curve_path), exist_ok=True)
    # with open(pr_curve_path, 'w') as f:
    #     f.write(json.dumps(pr_data))
    # print("pr curve save to ", pr_curve_path)
    print(config)
    End_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Start time: {}".format(start_time))
    print("End time: {}".format(End_time))

if __name__ == '__main__':
    main()
