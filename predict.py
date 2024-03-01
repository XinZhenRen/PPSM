import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models,unet
from Models import MSNet
from Models import EUNet
from Models import PraNet_Res2Net
from Models import HarDMSEG
from Models import U2Net
import progressbar
import time
from Metrics import performance_metrics

#python predict.py --model=FCBFormer --train-dataset=Kvasir --lastName=epoch19_mean0.9126776912994683_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=FCBFormer --train-dataset=Kvasir --lastName=epoch19_mean0.9126776912994683_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=FCBFormer --train-dataset=Kvasir --lastName=epoch19_mean0.9126776912994683_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=FCBFormer --train-dataset=Kvasir --lastName=epoch19_mean0.9126776912994683_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=FCBFormer --train-dataset=CVC --lastName=epoch19_mean0.9373621549762663_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=FCBFormer --train-dataset=CVC --lastName=epoch19_mean0.9373621549762663_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=FCBFormer --train-dataset=CVC --lastName=epoch19_mean0.9373621549762663_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=FCBFormer --train-dataset=CVC --lastName=epoch19_mean0.9373621549762663_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=FCBFormer --train-dataset=SUN --lastName=epoch19_mean0.8908551810393445_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=FCBFormer --train-dataset=SUN --lastName=epoch19_mean0.8908551810393445_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=FCBFormer --train-dataset=SUN --lastName=epoch19_mean0.8908551810393445_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=FCBFormer --train-dataset=SUN --lastName=epoch19_mean0.8908551810393445_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=FCBFormer --train-dataset=polyGen --lastName=epoch20_mean0.8531218968611111_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=UNet --train-dataset=Kvasir --lastName=epoch19_mean0.7166344218068116_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=UNet --train-dataset=Kvasir --lastName=epoch19_mean0.7166344218068116_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=UNet --train-dataset=Kvasir --lastName=epoch19_mean0.7166344218068116_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=UNet --train-dataset=Kvasir --lastName=epoch93_mean0.8611956337094306_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=UNet --train-dataset=CVC --lastName=epoch20_mean0.6250134989003181_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=UNet --train-dataset=CVC --lastName=epoch20_mean0.6250134989003181_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=UNet --train-dataset=CVC --lastName=epoch20_mean0.6250134989003181_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=UNet --train-dataset=CVC --lastName=epoch20_mean0.6250134989003181_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=UNet --train-dataset=SUN --lastName=epoch19_mean0.8381070806633835_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=UNet --train-dataset=SUN --lastName=epoch19_mean0.8381070806633835_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=UNet --train-dataset=SUN --lastName=epoch19_mean0.8381070806633835_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=UNet --train-dataset=SUN --lastName=epoch19_mean0.8381070806633835_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=UNet --train-dataset=polyGen --lastName=epoch20_mean0.5663443338823909_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=UNet --train-dataset=polyGen --lastName=epoch20_mean0.5663443338823909_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=UNet --train-dataset=polyGen --lastName=epoch20_mean0.5663443338823909_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=UNet --train-dataset=polyGen --lastName=epoch20_mean0.5663443338823909_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/



#python predict.py --model=U2Net --train-dataset=Kvasir --lastName=epoch17_mean0.7999835942297412_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=U2Net --train-dataset=Kvasir --lastName=epoch17_mean0.7999835942297412_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=U2Net --train-dataset=Kvasir --lastName=epoch17_mean0.7999835942297412_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=U2Net --train-dataset=Kvasir --lastName=epoch17_mean0.7999835942297412_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=U2Net --train-dataset=CVC --lastName=epoch20_mean0.7273024380650763_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=U2Net --train-dataset=CVC --lastName=epoch20_mean0.7273024380650763_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=U2Net --train-dataset=CVC --lastName=epoch20_mean0.7273024380650763_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=U2Net --train-dataset=CVC --lastName=epoch20_mean0.7273024380650763_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=U2Net --train-dataset=SUN --lastName=epoch19_mean0.879146451364904_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=U2Net --train-dataset=SUN --lastName=epoch19_mean0.879146451364904_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=U2Net --train-dataset=SUN --lastName=epoch19_mean0.879146451364904_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=U2Net --train-dataset=SUN --lastName=epoch19_mean0.879146451364904_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=U2Net --train-dataset=polyGen --lastName=epoch18_mean0.6723604523595186_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=U2Net --train-dataset=polyGen --lastName=epoch18_mean0.6723604523595186_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=U2Net --train-dataset=polyGen --lastName=epoch18_mean0.6723604523595186_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=U2Net --train-dataset=polyGen --lastName=epoch18_mean0.6723604523595186_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=ParNet --train-dataset=Kvasir --lastName=epoch14_mean0.8899620609357953_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=ParNet --train-dataset=Kvasir --lastName=epoch14_mean0.8899620609357953_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=ParNet --train-dataset=Kvasir --lastName=epoch14_mean0.8899620609357953_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=ParNet --train-dataset=Kvasir --lastName=epoch14_mean0.8899620609357953_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=ParNet --train-dataset=CVC --lastName=epoch20_mean0.9095898627257738_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=ParNet --train-dataset=CVC --lastName=epoch20_mean0.9095898627257738_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=ParNet --train-dataset=CVC --lastName=epoch20_mean0.9095898627257738_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=ParNet --train-dataset=CVC --lastName=epoch20_mean0.9095898627257738_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=ParNet --train-dataset=SUN --lastName=epoch19_mean0.8874973191693677_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=ParNet --train-dataset=SUN --lastName=epoch19_mean0.8874973191693677_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=ParNet --train-dataset=SUN --lastName=epoch19_mean0.8874973191693677_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=ParNet --train-dataset=SUN --lastName=epoch19_mean0.8874973191693677_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=ParNet --train-dataset=polyGen --lastName=epoch19_mean0.9060253759480958_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=ParNet --train-dataset=polyGen --lastName=epoch19_mean0.9060253759480958_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=ParNet --train-dataset=polyGen --lastName=epoch19_mean0.9060253759480958_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=ParNet --train-dataset=polyGen --lastName=epoch19_mean0.9060253759480958_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=MSNet --train-dataset=Kvasir --lastName=epoch9_mean0.886072740405798_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=MSNet --train-dataset=Kvasir --lastName=epoch9_mean0.886072740405798_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=MSNet --train-dataset=Kvasir --lastName=epoch9_mean0.886072740405798_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=MSNet --train-dataset=Kvasir --lastName=epoch9_mean0.886072740405798_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=MSNet --train-dataset=CVC --lastName=epoch16_mean0.9159295314648113_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=MSNet --train-dataset=CVC --lastName=epoch16_mean0.9159295314648113_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=MSNet --train-dataset=CVC --lastName=epoch16_mean0.9159295314648113_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=MSNet --train-dataset=CVC --lastName=epoch16_mean0.9159295314648113_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=MSNet --train-dataset=SUN --lastName=epoch17_mean0.8841503956846343_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=MSNet --train-dataset=SUN --lastName=epoch17_mean0.8841503956846343_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=MSNet --train-dataset=SUN --lastName=epoch17_mean0.8841503956846343_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=MSNet --train-dataset=SUN --lastName=epoch17_mean0.8841503956846343_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=MSNet --train-dataset=polyGen --lastName=epoch19_mean0.8662950731523208_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=MSNet --train-dataset=polyGen --lastName=epoch19_mean0.8662950731523208_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=MSNet --train-dataset=polyGen --lastName=epoch19_mean0.8662950731523208_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=MSNet --train-dataset=polyGen --lastName=epoch19_mean0.8662950731523208_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=HarDMSEG --train-dataset=Kvasir --lastName=epoch19_mean0.8846354703204977_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=HarDMSEG --train-dataset=Kvasir --lastName=epoch19_mean0.8846354703204977_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=HarDMSEG --train-dataset=Kvasir --lastName=epoch19_mean0.8846354703204977_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=HarDMSEG --train-dataset=Kvasir --lastName=epoch19_mean0.8846354703204977_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=HarDMSEG --train-dataset=CVC --lastName=epoch11_mean0.9060776223901843_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=HarDMSEG --train-dataset=CVC --lastName=epoch11_mean0.9060776223901843_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=HarDMSEG --train-dataset=CVC --lastName=epoch11_mean0.9060776223901843_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=HarDMSEG --train-dataset=CVC --lastName=epoch11_mean0.9060776223901843_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=HarDMSEG --train-dataset=SUN --lastName=epoch14_mean0.8860814116771487_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=HarDMSEG --train-dataset=SUN --lastName=epoch14_mean0.8860814116771487_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=HarDMSEG --train-dataset=SUN --lastName=epoch14_mean0.8860814116771487_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=HarDMSEG --train-dataset=SUN --lastName=epoch14_mean0.8860814116771487_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=HarDMSEG --train-dataset=polyGen --lastName=epoch13_mean0.8821159657524977_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=HarDMSEG --train-dataset=polyGen --lastName=epoch13_mean0.8821159657524977_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=HarDMSEG --train-dataset=polyGen --lastName=epoch13_mean0.8821159657524977_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=HarDMSEG --train-dataset=polyGen --lastName=epoch13_mean0.8821159657524977_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=EUNet --train-dataset=Kvasir --lastName=epoch15_mean0.8741291210574832_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=EUNet --train-dataset=Kvasir --lastName=epoch15_mean0.8741291210574832_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=EUNet --train-dataset=Kvasir --lastName=epoch15_mean0.8741291210574832_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=EUNet --train-dataset=Kvasir --lastName=epoch15_mean0.8741291210574832_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


#python predict.py --model=EUNet --train-dataset=CVC --lastName=epoch15_mean0.8849822087366073_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=EUNet --train-dataset=CVC --lastName=epoch15_mean0.8849822087366073_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=EUNet --train-dataset=CVC --lastName=epoch15_mean0.8849822087366073_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=EUNet --train-dataset=CVC --lastName=epoch15_mean0.8849822087366073_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=EUNet --train-dataset=SUN --lastName=epoch14_mean0.873147104474339_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/

#python predict.py --model=EUNet --train-dataset=polyGen --lastName=epoch11_mean0.7362021543747426_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict.py --model=EUNet --train-dataset=polyGen --lastName=epoch11_mean0.7362021543747426_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict.py --model=EUNet --train-dataset=polyGen --lastName=epoch11_mean0.7362021543747426_devicecuda --test-dataset=SUN --data-root=./SUN/
#python predict.py --model=EUNet --train-dataset=polyGen --lastName=epoch11_mean0.7362021543747426_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.test_dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "SUN":
        img_path = args.root + "Frame/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "GT/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "polyGen":
        img_path = args.root + "imagesAll_positive/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "imagesAll_mask/*"
        target_paths = sorted(glob.glob(depth_path))
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1
    )

    _, test_indices, _ = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()
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
            args.batch_size = 1
        case "ParNet":
            model = PraNet_Res2Net.PraNet()
        case "HarDMSEG":
            model = HarDMSEG.HarDMSEG()
        case "U2Net":
            model = U2Net.u2net_full()
    state_dict = torch.load(
        "./Trained_models/{}_{}_{}.pt".format(args.model,args.train_dataset,args.lastName)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/{}".format(args.model)):
        os.makedirs("./Predictions/{}".format(args.model))
    if not os.path.exists("./Predictions/{}/Trained on {}/".format(args.model,args.train_dataset)):
        os.makedirs("./Predictions/{}/Trained on {}/".format(args.model,args.train_dataset))
    if not os.path.exists(
                "./Predictions/{}/Trained on {}/test_on_{}".format(args.model,args.train_dataset,args.test_dataset)):
                os.makedirs(
                "./Predictions/{}/Trained on {}/test_on_{}".format(args.model,args.train_dataset,args.test_dataset)
                )
    imageWritedir="./Predictions/{}/Trained on {}/test_on_{}".format(args.model,args.train_dataset,args.test_dataset)

    # if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
    #     os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
    # if not os.path.exists(
    #     "./Predictions/Trained on {}/Tested on {}".format(
    #         args.train_dataset, args.test_dataset
    #     )
    # ):
    #     os.makedirs(
    #         "./Predictions/Trained on {}/Tested on {}".format(
    #             args.train_dataset, args.test_dataset
    #         )
    #     )

    t = time.time()
    t1=0
    model.eval()
    perf_accumulator = []
    print("device:",device)

    for i, (data, target) in enumerate(test_dataloader):
        t0=time.time()
        data, target = data.to(device), target.to(device)
        output = model(data)
        t1 = t1 + time.time() - t0
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0

        perf_accumulator.append(perf_measure(output, target).item())


        cv2.imwrite(imageWritedir+"/{}.png".format(os.path.splitext(os.path.basename(target_paths[i]))[0]),predicted_map * 255)
        # cv2.imwrite(
        # "./Predictions/{}/Trained on {}/Tested on {}/{}".format(
        #     args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
        # ),
        # predicted_map * 255,
        # )
        if i + 1 < len(test_dataloader) :
            if i%100!=0:
                pass
            else:
                print(
                    "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                        i + 1,
                        len(test_dataloader),
                        100.0 * (i + 1) / len(test_dataloader),
                        np.mean(perf_accumulator),
                        time.time() - t,
                    ),
                    end="",
                )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}\tFPS:{:.3f} TFPS:{:.3f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                    len(test_dataloader)/(time.time() - t),
                    len(test_dataloader)/t1
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=["FCBFormer", "UNet","UNet++","MSNet","ParNet","EUNet","HarDMSEG","U2Net"]
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC","SUN","polyGen"]
    )
    parser.add_argument(
        "--lastName", type=str
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC","SUN","polyGen"]
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()
