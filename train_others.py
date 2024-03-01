import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
from Models import unet
from Models import MSNet
from Models import EUNet
from Models import PraNet_Res2Net
from Models import HarDMSEG
from Models import U2Net
import progressbar
import time

#import tqdm as tqdm

#python train_others.py --model=UNet --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=UNet --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=UNet --dataset=SUN --data-root=./SUN/ --resume-training=./Trained_models/FCBFormer_SUN_epoch13_mean0.903108070814724_devicecuda.pt
#python train_others.py --model=UNet --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python train_others.py --model=FCBFormer --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=FCBFormer --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=FCBFormer --dataset=SUN --data-root=./SUN/
#python train_others.py --model=FCBFormer --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/ --resume-training=./Trained_models/FCBFormer_polyGen_epoch8_mean0.7508144212738539_devicecuda.pt --fromEpoch=8


#python train_others.py --model=MSNet --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=MSNet --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=MSNet --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#python train_others.py --model=MSNet --dataset=SUN --data-root=./SUN/

#python train_others.py --model=EUNet --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=EUNet --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=EUNet --dataset=SUN --data-root=./SUN/ --resume-training=./Trained_models/EUNet_SUN_epoch14_mean0.873147104474339_devicecuda.pt --fromEpoch=14
#python train_others.py --model=EUNet --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python train_others.py --model=ParNet --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=ParNet --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=ParNet --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#python train_others.py --model=ParNet --dataset=SUN --data-root=./SUN/


#python train_others.py --model=HarDMSEG --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=HarDMSEG --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=HarDMSEG --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#python train_others.py --model=HarDMSEG --dataset=SUN --data-root=./SUN/

#python train_others.py --model=U2Net --dataset=Kvasir --data-root=./kvasir-seg/
#python train_others.py --model=U2Net --dataset=CVC --data-root=./CVC-ClinicDB/
#python train_others.py --model=U2Net --dataset=SUN --data-root=./SUN/
#python train_others.py --model=U2Net --dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    try:
        t = time.time()
        model.train()
        loss_accumulator = []
        bar_epoch = progressbar.ProgressBar(widgetmodel =[
            progressbar.Timer(),
            progressbar.Percentage(),
            '(', progressbar.SimpleProgress(), ')',
            '(', progressbar.ETA(), ')',
            '(', progressbar.AdaptiveETA(), ')'])
        for batch_idx, (data, target) in enumerate(train_loader):
            #without try#########################################################
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
            loss.backward()
            optimizer.step()
            loss_accumulator.append(loss.item())
            # if batch_idx + 1 < len(train_loader):
            #     print(
            #         "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
            #             epoch,
            #             (batch_idx + 1) * len(data),
            #             len(train_loader.dataset),
            #             100.0 * (batch_idx + 1) / len(train_loader),
            #             loss.item(),
            #             time.time() - t,
            #         ),
            #         end="",
            #     )
            # else:
            #     print(
            #         "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
            #             epoch,
            #             (batch_idx + 1) * len(data),
            #             len(train_loader.dataset),
            #             100.0 * (batch_idx + 1) / len(train_loader),
            #             np.mean(loss_accumulator),
            #             time.time() - t,
            #         )
            #     )

            # try:#########################################################################################
            #     data, target = data.to(device), target.to(device)
            #     optimizer.zero_grad()
            #     output = model(data)
            #     loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
            #     loss.backward()
            #     optimizer.step()
            #     loss_accumulator.append(loss.item())
            #     if batch_idx + 1 < len(train_loader):
            #         print(
            #             "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
            #                 epoch,
            #                 (batch_idx + 1) * len(data),
            #                 len(train_loader.dataset),
            #                 100.0 * (batch_idx + 1) / len(train_loader),
            #                 loss.item(),
            #                 time.time() - t,
            #             ),
            #             end="",
            #         )
            #     else:
            #         print(
            #             "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
            #                 epoch,
            #                 (batch_idx + 1) * len(data),
            #                 len(train_loader.dataset),
            #                 100.0 * (batch_idx + 1) / len(train_loader),
            #                 np.mean(loss_accumulator),
            #                 time.time() - t,
            #             )
            #         )
            # except Exception as e:
            #     print("!some err in train_epoch but continue:",e)
            #     continue
    except Exception as e:
        print("!!some err in train_epoch next but continue:", e)
        print("\nbatch_idx",batch_idx)
        print("train_loader.dataset.__getitem__()", train_loader.dataset.__getitem__(batch_idx))
    return np.mean(loss_accumulator)

def train(args):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    if hasattr(torch.cuda,'empty_cache'):
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:" )
    print(device )
    if args.dataset == "Kvasir":
        img_path = args.root + "images/*"
        print("img_path:",img_path)
        input_paths = sorted(glob.glob(img_path))#glob.glob 遍历文件
        print("input_paths:", input_paths)
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "SUN":
        img_path = args.root + "Frame/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "GT/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "polyGen":
        img_path = args.root + "imagesAll_positive/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "imagesAll_mask/*"
        target_paths = sorted(glob.glob(depth_path))

    match args.model:
        case "FCBFormer":
            model = models.FCBFormer()
        case "UNet":
            model = unet.UNet()
        case "UNet++":
            model = unet.UNet()
        case "MSNet":
            model = MSNet.MSNet()
        case "EUNet":
            model = EUNet.EUNet()
            args.batch_size=2
        case "ParNet":
            model = PraNet_Res2Net.PraNet()
        case "HarDMSEG":
            model = HarDMSEG.HarDMSEG()
        case "U2Net":
            model = U2Net.u2net_full()


    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()

    perf = performance_metrics.DiceScore()


    if args.mgpu == "true":
        model = nn.DataParallel(model)

    if args.resumePath != None:
        model.load_state_dict(torch.load(args.resumePath)['model_state_dict'])

        #Nepoch=torch.load(args.resumePath)['epoch']
        Nepoch = args.fromEpoch
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        torch.optim.AdamW.load_state_dict(optimizer,state_dict=torch.load(args.resumePath)['optimizer_state_dict'])
        for state in optimizer.state.values():#put optim to cuda
            for k,v in state.items():
                if isinstance(v,torch.Tensor):
                    state[k]=v.cuda()

        model.to(device)
    else:
        model.to(device)
        Nepoch = 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print("model in :", next(model.parameters()).device)


    try:
        if not os.path.exists("./Trained_models"):#make dirs if there is not dirs
            os.makedirs("./Trained_models")

        prev_best_test = None#
        if args.lrs == "true":#learning-rate-scheduler
            if args.lrs_min > 0:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
                )
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, verbose=True
                )

        print("now model start training from epoch:", Nepoch)
        print("now model start training with epoch:", args.epochs)

        bar=progressbar.ProgressBar(widgets=[
            progressbar.Timer(),
            progressbar.Percentage(),
        '(',progressbar.SimpleProgress(),')',
        '(', progressbar.ETA(), ')',
        '(',progressbar.AbsoluteETA(),')'])

        for epoch in bar(range(Nepoch, args.epochs + 1)):
            try:
                loss = train_epoch(
                    model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss
                )
                test_measure_mean, test_measure_std = test(
                    model, device, val_dataloader, epoch, perf
                )
            except KeyboardInterrupt:
                print("Training interrupted by user")
                sys.exit(0)
            if args.lrs == "true":
                scheduler.step(test_measure_mean)
            if prev_best_test == None or test_measure_mean > prev_best_test:
                print("Saving...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict()
                        if args.mgpu == "false"
                        else model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_measure_mean": test_measure_mean,
                        "test_measure_std": test_measure_std,
                    },
                    "Trained_models/{}_".format(args.model) + args.dataset +"_epoch"+str(epoch)+"_mean"+str(test_measure_mean)+"_device"+str(device)+ ".pt",
                )
                prev_best_test = test_measure_mean
    except:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict()
                if args.mgpu == "false"
                else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "test_measure_mean": test_measure_mean,
                "test_measure_std": test_measure_std,
            },
            "Trained_models/{}_".format(args.model) + args.dataset +"_epoch"+str(epoch)+"_mean"+str(test_measure_mean)+"_device"+str(device)+".pt",
        )

    print("prev_best_test", prev_best_test)

def get_args():
    parser = argparse.ArgumentParser(description="Train on specified dataset")
    parser.add_argument(
        "--model", type=str, required=True, choices=["FCBFormer", "UNet","UNet++","MSNet","ParNet","EUNet","HarDMSEG","U2Net"]
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC","SUN","polyGen"])
    parser.add_argument("--data-root", type=str, default="./SUN/",required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument('--fromEpoch', type=int, default=1)


    parser.add_argument('--video_time_clips', type=int, default=5)
    parser.add_argument('--resume-training', type=str, default=None, dest="resumePath")


    return parser.parse_args()







def main():
    args = get_args()

    train(args)


if __name__ == "__main__":
    main()
