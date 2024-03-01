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
import progressbar
import time
from Metrics import performance_metrics

#python predict-prompt.py --model=PPSN --train-dataset=Kvasir --lastName=epoch16_mean0.9804620838165283_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --train-dataset=polyGen --lastName=epoch20_mean1.1576705071678821_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#python predict-prompt.py --model=PPSN --train-dataset=Kvasir --lastName=epoch16_mean0.9804620838165283_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --train-dataset=CVC --lastName=epoch15_mean0.9923643225529155_devicecuda --test-dataset=Kvasir --data-root=./kvasir-seg/
#python predict-prompt.py --model=PPSN --train-dataset=CVC --lastName=epoch15_mean0.9923643225529155_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/
#python predict-prompt.py --model=PPSN --train-dataset=CVC --lastName=epoch77_mean0.9372267714260664_devicecuda --test-dataset=CVC --data-root=./CVC-ClinicDB/
#python predict-prompt.py --model=PPSN --train-dataset=polyGen --lastName=epoch20_mean1.1576705071678821_devicecuda --test-dataset=polyGen --data-root=./polypGen2021_MultiCenterData_v3/




def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:")
    print(device)

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
    prompt_path=args.root

    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths,prompt=prompt_path, batch_size=1
    )

    _, test_indices, _ = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()
    match args.model:
        case "FCBFormer":
            model = models.FCBFormer()
        case "UNet":
            model = unet.UNet()
        case "PPSN":
            model = models.PPSN(chooseProb=args.p)
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
    if not os.path.exists("./Predictions/{}/Trained on {}/".format(args.model, args.train_dataset)):
        os.makedirs("./Predictions/{}/Trained on {}/".format(args.model, args.train_dataset))
    if not os.path.exists("./Predictions/{}/Trained on {}/p{}/".format(args.model, args.train_dataset,args.p)):
        os.makedirs("./Predictions/{}/Trained on {}/p{}/".format(args.model, args.train_dataset,args.p))
    if not os.path.exists(
            "./Predictions/{}/Trained on {}/p{}/test_on_{}".format(args.model, args.train_dataset,args.p,args.test_dataset)):
        os.makedirs(
            "./Predictions/{}/Trained on {}/p{}/test_on_{}".format(args.model, args.train_dataset,args.p,args.test_dataset)
        )
    imageWritedir = "./Predictions/{}/Trained on {}/p{}/test_on_{}".format(args.model, args.train_dataset,args.p,
                                                                       args.test_dataset)

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
    t1 = 0
    model.eval()
    perf_accumulator = []
    print("device:", device)

    for i, (data, prompt,target) in enumerate(test_dataloader):
        t0 = time.time()
        data,prompt, target = data.to(device),prompt.to(device), target.to(device)
        output = model(data,prompt)
        t1 = t1 + time.time() - t0
        perf_accumulator.append(perf_measure(output,target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0

        cv2.imwrite(imageWritedir + "/{}.png".format(os.path.splitext(os.path.basename(target_paths[i]))[0]),
                    predicted_map * 255)
        # cv2.imwrite(
        # "./Predictions/{}/Trained on {}/Tested on {}/{}".format(
        #     args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
        # ),
        # predicted_map * 255,
        # )
        if i + 1 < len(test_dataloader):
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
        "--model", type=str, required=True, choices=["FCBFormer", "UNet","PPSN"]
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
    parser.add_argument(
        "--p", type=float, default=1)
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()
