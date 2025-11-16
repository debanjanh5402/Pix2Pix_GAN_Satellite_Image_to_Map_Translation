import torch
import os, math, json
from torch import nn
from torch.utils import data
from torch.optim import Optimizer
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


###################################################################################################
# ---------------------------------     Paired Dataloader     -------------------------------------
###################################################################################################
class PairedImageDatasetLoader(data.Dataset):
    def __init__(self, root_dir: str, resize_size: int, split_width: int):
        self.image_filenames = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
        self.image_paths = [os.path.join(root_dir, f) for f in sorted(self.image_filenames)]
        self.split_width = split_width
        self.transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        image = default_loader(path)
        # guard: ensure split_width is within image width
        sw = min(self.split_width, image.width)
        input_image = image.crop((0, 0, sw, image.height))
        target_image = image.crop((sw, 0, image.width, image.height))
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)
        return input_tensor, target_tensor
    

###################################################################################################
# ----------------------------------     Visulaising dataset     ----------------------------------
###################################################################################################
def visualize_dataset(dataloader:data.DataLoader, num_samples:int=16, cols:int=4, subplot_width:int=4, title:str=None):
    batch_size = dataloader.batch_size; num_batch = math.ceil(num_samples/batch_size)
    input_batch_list, target_batch_list = [], []
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= num_batch:
            break
        input_batch_list.append(input_batch)
        target_batch_list.append(target_batch)

    input_image_list = torch.cat(input_batch_list, dim=0)
    target_image_list = torch.cat(target_batch_list, dim=0)

    total_plots = num_samples*2
    rows = math.ceil(total_plots/cols)
    plt.figure(figsize=(cols*subplot_width, rows*subplot_width), dpi=300)
    for i in range(num_samples):

        plt.subplot(rows, cols, (2*i)+1)
        input_image = input_image_list[i].permute(1,2,0)
        input_image = (input_image+1)/2
        plt.imshow(input_image)
        plt.title("Input"); plt.axis("off")
        plt.subplot(rows, cols, (2*i)+2)
        target_image = target_image_list[i].permute(1,2,0)
        target_image = (target_image+1)/2
        plt.imshow(target_image)
        plt.title("Target"); plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return None


###################################################################################################
# ------------------------------------     Saving Model     ---------------------------------------
###################################################################################################
def save_model(model:nn.Module, optimizer:Optimizer, epoch:int, save_path:str) -> None:
    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict()
    }, save_path)
    return None


###################################################################################################
# -----------------------------------     Saving History     --------------------------------------
###################################################################################################
def save_history(history:Dict[str, List], save_path:str) -> None:
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    return None


###################################################################################################
# ---------------------------------     Plotting History     -------------------------------------
###################################################################################################
def plot_history(history:Dict[str, List]) -> None:
    d_loss, g_loss = history["D_loss"], history["G_loss"]
    g_adv_loss, g_recon_loss = history["G_adv_loss"], history["G_recon_loss"]

    epoch = range(1, len(d_loss)+1)
    plots = [d_loss, g_loss, g_adv_loss, g_recon_loss]
    titles = ["Discriminator Loss", "Generator Loss", "Generator Adversarial Loss", "Generator Reconstruction Loss"]

    plt.figure(figsize=(20, 12), dpi=300)

    for i in range(len(plots)):
        plt.subplot(2, 2, i+1)
        plt.plot(epoch, plots[i])
        plt.scatter(epoch, plots[i])
        plt.title(titles[i])
        plt.xlabel("Epoch")
        plt.grid()
    
    plt.tight_layout()
    plt.show()