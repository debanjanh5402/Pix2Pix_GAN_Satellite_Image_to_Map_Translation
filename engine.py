import torch
from torch import nn
from torch.utils import data
from torch import optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure #type:ignore
from typing import Tuple, Dict
from tqdm.auto import tqdm #type:ignore
import matplotlib.pyplot as plt


###################################################################################################
# ----------------------------     Discriminator Training Step     --------------------------------
###################################################################################################
def train_discriminator_step(generator:nn.Module, discriminator:nn.Module, real_input:torch.tensor, real_target:torch.tensor,
                             loss_fn_adv:nn.Module, optimizer_D:optim.Optimizer, device:torch.device) -> float:
    
    discriminator.train(); generator.eval()

    optimizer_D.zero_grad()

    gen_output = generator(real_input).detach()
    D_real_output = discriminator(real_input, real_target)
    D_fake_output = discriminator(real_input, gen_output)

    all_ones = torch.ones_like(D_real_output, device=device)
    all_zeros = torch.zeros_like(D_fake_output, device=device)

    D_real_loss = loss_fn_adv(D_real_output, all_ones)
    D_fake_loss = loss_fn_adv(D_fake_output, all_zeros)
    D_loss = (D_real_loss + D_fake_loss)/2

    D_loss.backward()

    optimizer_D.step()

    return D_loss.item()


###################################################################################################
# ------------------------------     Generator Training Step     ----------------------------------
###################################################################################################
def train_generator_step(generator:nn.Module, discriminator:nn.Module, real_input:torch.tensor, real_target:torch.tensor,
                         loss_fn_adv:nn.Module, loss_fn_recon:nn.Module, lambda_gen:float,
                         optimizer_G:optim.Optimizer, device:torch.device) -> Tuple[float, float, float]:
    
    generator.train(); discriminator.eval()

    optimizer_G.zero_grad()

    fake_target = generator(real_input)

    D_fake_pred = discriminator(real_input, fake_target)
    real_targets = torch.ones_like(D_fake_pred, device=device)
    loss_g_adv = loss_fn_adv(D_fake_pred, real_targets)

    loss_g_recon = loss_fn_recon(fake_target, real_target)

    G_loss = loss_g_adv + lambda_gen * loss_g_recon
    G_loss.backward()

    optimizer_G.step()

    return G_loss.item(), loss_g_adv.item(), loss_g_recon.item()


###################################################################################################
# ----------------------------------     Generate Images     --------------------------------------
###################################################################################################
def generate_images(val_dataloader: data.DataLoader, generator: nn.Module, device: torch.device, epoch:int):
    generator.eval()
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    with torch.no_grad():
        input_image, target_image = next(iter(val_dataloader))
        input_image, target_image = input_image.to(device), target_image.to(device)
        pred_image = generator(input_image)
        input_image_norm = (input_image + 1) / 2; target_image_norm = (target_image + 1) / 2; pred_image_norm = (pred_image + 1) / 2
        psnr = psnr_metric(pred_image_norm, target_image_norm)
        ssim = ssim_metric(pred_image_norm, target_image_norm)

    input_sample = input_image_norm[0].permute(1, 2, 0).cpu().numpy()
    target_sample = target_image_norm[0].permute(1, 2, 0).cpu().numpy()
    pred_sample = pred_image_norm[0].permute(1, 2, 0).cpu().numpy()

    # Plotting
    plt.figure(figsize=(18, 6), dpi=300)
    
    plt.subplot(131)
    plt.imshow(input_sample); plt.title("Input image"); plt.axis("off")

    plt.subplot(132)
    plt.imshow(target_sample); plt.title("Target image"); plt.axis("off")

    plt.subplot(133)
    # Use .item() to get scalar for printing
    plt.imshow(pred_sample); plt.title(f"Pred Image\nSSIM:{ssim.item():.4f}, PSNR:{psnr.item():.4f}"); plt.axis("off")

    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()

    plt.show()

    return None

###################################################################################################
# ---------------------------------     GAN Training Step     -------------------------------------
###################################################################################################
def train_GAN(generator:nn.Module, discriminator:nn.Module, train_dataloader:data.DataLoader, val_dataloader:data.DataLoader,
              loss_fn_adv:nn.Module, loss_fn_recon:nn.Module, optimizer_G:optim.Optimizer, optimizer_D:optim.Optimizer,
              lambda_gen:float, epochs:int, device:torch.device) -> Dict[str, list]:
    
    results = {"G_loss":[], "D_loss":[], "G_adv_loss":[], "G_recon_loss":[]}

    generator.to(device); discriminator.to(device)

    for epoch in tqdm(range(epochs), total=epochs):
        G_loss_total, D_loss_total, G_adv_loss_total, G_recon_loss_total = 0, 0, 0, 0

        generator.train(); discriminator.train()

        for real_input, real_target in tqdm(train_dataloader, total=len(train_dataloader)):
            real_input, real_target = real_input.to(device), real_target.to(device)

            D_loss = train_discriminator_step(generator, discriminator, real_input, real_target, loss_fn_adv, optimizer_D, device)

            G_loss, G_adv_loss, G_recon_loss = train_generator_step(generator, discriminator, real_input, real_target, 
                                                                    loss_fn_adv, loss_fn_recon, lambda_gen, optimizer_G, device)
            
            D_loss_total += D_loss; G_loss_total += G_loss
            G_adv_loss_total += G_adv_loss; G_recon_loss_total += G_recon_loss

        avg_D_loss = D_loss_total/len(train_dataloader); avg_G_loss = G_loss_total/len(train_dataloader)
        avg_G_adv_loss = G_adv_loss_total/len(train_dataloader); avg_G_recon_loss = G_recon_loss_total/len(train_dataloader)

        print(f"|| Epoch {epoch+1} Summary ||")
        print(f"D_loss:{avg_D_loss:.4f} | G_loss:{avg_G_loss:.4f} | G_adv_loss:{avg_G_adv_loss:.4f} | G_recon_loss:{avg_G_recon_loss:.4f}")

        results["D_loss"].append(avg_D_loss); results["G_loss"].append(avg_G_loss)
        results["G_adv_loss"].append(avg_G_adv_loss); results["G_recon_loss"].append(avg_G_recon_loss)

        if (epoch+1)%5==0:
            generate_images(val_dataloader, generator, device, epoch=(epoch+1))


    return results