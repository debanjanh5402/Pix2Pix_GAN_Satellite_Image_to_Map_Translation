import math
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio # type:ignore
from typing import Dict, List
from tqdm.auto import tqdm # type:ignore
import matplotlib.pyplot as plt


###################################################################################################
# -----------------------------     SSIM & PSNR calculations     ---------------------------------
###################################################################################################
def ssim_psnr_results(model:nn.Module, dataloader:DataLoader, device:torch.device) -> Dict:
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    batch_ssim_list, batch_psnr_list = [], []
    with torch.no_grad():
        for input_batch, target_batch in tqdm(dataloader, total=len(dataloader)):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            pred_batch = model(input_batch)

            target_batch = (target_batch + 1)/2
            pred_batch = (pred_batch + 1)/2

            batch_ssim = ssim_metric(pred_batch, target_batch).item()
            batch_psnr = psnr_metric(pred_batch, target_batch).item()

            batch_ssim_list.append(batch_ssim)
            batch_psnr_list.append(batch_psnr)
    
    mean_ssim = sum(batch_ssim_list)/len(batch_ssim_list)
    std_dev_ssim = math.sqrt(sum([(x-mean_ssim)**2 for x in batch_ssim_list])/len(batch_ssim_list))

    mean_psnr = sum(batch_psnr_list)/len(batch_psnr_list)
    std_dev_psnr = math.sqrt(sum([(x-mean_psnr)**2 for x in batch_psnr_list])/len(batch_psnr_list))

    results = {
        "ssim_scores": batch_ssim_list,
        "psnr_scores": batch_psnr_list,
        "mean_ssim": mean_ssim,
        "mean_psnr": mean_psnr,
        "std_dev_ssim": std_dev_ssim,
        "std_dev_psnr": std_dev_psnr
    }

    print("="*75)
    print("Results")
    print("="*75)
    print(f"Mean SSIM: {mean_ssim*100:0.2f}% ± {std_dev_ssim*100:0.2f}%")
    print(f"Mean PSNR: {mean_psnr:0.2f} dB ± {std_dev_psnr:0.2f} dB")

    return results


###################################################################################################
# -----------------------------     SSIM & PSNR distribution     ---------------------------------
###################################################################################################
def plot_result_distribution(results:Dict):
    ssim_scores = results["ssim_scores"]; psnr_scores=results["psnr_scores"]
    mean_ssim = results["mean_ssim"]; mean_psnr = results["mean_psnr"]
    std_dev_ssim = results["std_dev_ssim"]; std_dev_psnr = results["std_dev_psnr"]

    indices = range(1, len(ssim_scores)+1)

    plt.figure(figsize=(20, 6), dpi=300)
    
    plt.subplot(121)
    plt.scatter(indices, ssim_scores, color="royalblue", s=10)
    plt.title("SSIM scores"); plt.grid()
    plt.xlabel("Indices"); plt.ylabel("SSIM scores")
    plt.ylim(0.2,1)
    plt.axhline(y=mean_ssim, label="μ", color="green", ls="--")
    plt.axhline(y=mean_ssim+(3*std_dev_ssim), label="μ + 3σ", color="red", ls="--")
    plt.axhline(y=mean_ssim-(3*std_dev_ssim), label="μ - 3σ", color="red", ls="--")
    plt.legend()

    plt.subplot(122)
    plt.scatter(indices, psnr_scores, color="darkorange", s=10)
    plt.title("PSNR scores"); plt.grid()
    plt.xlabel("Indices"); plt.ylabel("PSNR scores")
    plt.axhline(y=mean_psnr, label="μ", color="green", ls="--")
    plt.axhline(y=mean_psnr+(3*std_dev_psnr), label="μ + 3σ", color="red", ls="--")
    plt.axhline(y=mean_psnr-(3*std_dev_psnr), label="μ - 3σ", color="red", ls="--")
    plt.legend()

    return None

###################################################################################################
# ------------------------------------     Predictions     ----------------------------------------
###################################################################################################
def predict_few_samples(dataloader:DataLoader, model:nn.Module, num_samples:int, device:torch.device, 
                        cols:int=6, subplot_width:int=4, title:str=None):
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    batch_size = dataloader.batch_size
    num_batch = math.ceil(num_samples/batch_size)

    input_batch_list, target_batch_list, predicted_batch_list = [], [], []
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(dataloader):
            if i>= num_batch:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            predicted_batch = model(input_batch)

            input_batch_list.append(input_batch)
            target_batch_list.append(target_batch)
            predicted_batch_list.append(predicted_batch)

    input_image_list = torch.cat(input_batch_list, dim=0)
    target_image_list = torch.cat(target_batch_list, dim=0)
    predicted_image_list = torch.cat(predicted_batch_list, dim=0)

    total_plots=num_samples*3
    rows = math.ceil(total_plots/cols)
    plt.figure(figsize=(cols*subplot_width, rows*subplot_width+1), dpi=300)

    for i in range(num_samples):
        plt.subplot(rows, cols, (3*i)+1)
        input_image = input_image_list[i].permute(1, 2, 0)
        input_image = (input_image+1)/2
        plt.imshow(input_image.cpu().numpy())
        plt.title("Input"); plt.axis("off")

        plt.subplot(rows, cols, (3*i)+2)
        target_image = target_image_list[i]
        target_image = (target_image+1)/2
        plt.imshow(target_image.permute(1, 2, 0).cpu().numpy())
        plt.title("Target"); plt.axis("off")

        plt.subplot(rows, cols, (3*i)+3)
        predicted_image = predicted_image_list[i]
        predicted_image = (predicted_image+1)/2
        plt.imshow(predicted_image.permute(1, 2, 0).cpu().numpy())

        ssim = ssim_metric(predicted_image.unsqueeze(dim=0), target_image.unsqueeze(dim=0)).item()
        psnr = psnr_metric(predicted_image.unsqueeze(dim=0), target_image.unsqueeze(dim=0)).item()

        plt.title(f"Prediction\n(SSIM: {ssim*100:0.2f}%, PSNR: {psnr:0.2f} dB)"); plt.axis("off")
    
    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
    return None