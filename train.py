import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os,sys,datetime
import argparse

from celeba_dataloader import *
from network import *
from light_cnn import LightCNN_29Layers_v2
torch.backends.cudnn.bencmark = True


def save_model(folder, model, filename, re_model, re_filename):
    os.makedirs(folder,exist_ok=True)
    torch.save(model.state_dict(), folder + filename)
    torch.save(re_model.state_dict(), folder + re_filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def main():
    parser = argparse.ArgumentParser(description='Net (16 * 16) --> (32 * 32) and (64 * 64)')
    parser.add_argument('--lr', default=5e-5, type=float, help='Restoration network learning rate')
    parser.add_argument('--Re_lr', default=0.0001, type=float, help='Recognition network learning rate')
    parser.add_argument('--bs', default=64, type=int, help='Batch size')
    parser.add_argument('--epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--n_epochs', default=15, type=int, help='The number of epochs')
    parser.add_argument('--n_gpu', type=int, default=1, help='The number of GPUs to use')
    args = parser.parse_args()
    print(args)
    use_cuda = torch.cuda.is_available()

    print('start: time={}'.format(dt()))
    # Dataset
    data_transforms = {
        'train_input':
        transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val_input':
        transforms.Compose([
            transforms.ToTensor(),
        ])
    }

    
    celebAData_val = CelebADataset(
        data_transforms['val_input'], 'val', True, 1)
    
    dataloader_val = DataLoader(
        celebAData_val, batch_size=8, shuffle=True, num_workers=1)

    model = Net()
    re_net = LightCNN_29Layers_v2()
    criterion_n1 = nn.L1Loss()
        
    model = model.cuda(cuda_device_)
    model.train()
    re_net = re_net.cuda(cuda_device_)
    re_net.train()
    criterion_n1 = criterion_n1.cuda(cuda_device_)
    criterion_re = nn.CrossEntropyLoss().cuda(cuda_device_)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 25], gamma=0.5)
    print('G opt :\t{}'.format(optimizer.__dict__['defaults']))

    optimizer_Re = optim.Adam(
                params=[
                    {'params':list(re_net.parameters())[:-5]}, # common model
                    {'params':list(re_net.parameters())[-2:]}, # fusion conv2d
                    {'params': re_net.fc.weight, 'lr':args.Re_lr * 10}, 
                    {'params': re_net.fc.bias, 'lr':args.Re_lr * 20},
                    {'params': re_net.fc2.weight, 'lr':args.Re_lr * 10}
                ],
                lr=args.Re_lr, 
                betas=(0.9,0.999), 
                eps=1e-8)
    Re_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_Re, milestones=[5, 10, 15, 18, 24, 30], gamma=0.1)
    print('Re opt:\t{}'.format(optimizer_Re.__dict__['defaults']))

    for epoch in range(args.epoch, args.n_epochs):
        lr_scheduler.step()
        Re_lr_scheduler.step()
        if epoch % 5 == 0:
            print('lr: {}'.format(lr_scheduler.get_lr()))
            print('re_lr: {}'.format(Re_lr_scheduler.get_lr()))

        total = 0
        correct = 0

        celebAData_train = CelebADataset(data_transforms['train_input'])
        dataloader_train = DataLoader(
            celebAData_train,
            batch_size=args.bs,
            shuffle=True,
            num_workers=0)
        for i, (label, img, img_cropped) in enumerate(dataloader_train):
            if use_cuda: label, img, img_cropped = label.cuda(cuda_device_), img.cuda(cuda_device_), img_cropped.cuda(cuda_device_)
            
            out, selected_feature = model(img)

            # Train Recognition
            lambda_diff = 1
            outputs_diff, _ = re_net((out.detach(), selected_feature.detach()))
            outputs_same, _ = re_net((img_cropped, selected_feature.detach()))
            loss_diff = criterion_re(outputs_diff, label[:,0])
            loss_same = criterion_re(outputs_same, label[:,0])
            lossd_same = loss_same.item() 
            lossd_diff = loss_diff.item()
            loss_re = loss_same + lambda_diff * loss_diff
            
            optimizer_Re.zero_grad()
            loss_re.backward()
            optimizer_Re.step()

            _, predicted = torch.max(outputs_same.data, 1)
            total += label.shape[0]
            correct += predicted.eq(label.reshape(label.shape[0]).data).sum().cpu()


            # Train Restoration
            loss_x4 = criterion_n1(out, img_cropped)
            outputs_diff, _ = re_net((out, selected_feature))
            loss_r = criterion_re(outputs_diff, label[:,0])
            loss = loss_x4 + loss_r * 0.005
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        save_model('saved', model, 'res_net_{}.pth'.format(epoch), re_net, 'rec_net_{}.pth'.format(epoch))
    print('finish: time={}'.format(dt()))

if __name__ == '__main__':
    cuda_device_ = 0
    main()
