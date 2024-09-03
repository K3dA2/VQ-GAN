import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import os
import lpips
from model.vqvae import VQVAE
from discriminator import DiscriminatorLoss
from utils import get_data_loader, count_parameters, save_img_tensors_as_grid
import torch.nn.functional as F

def gan_weight(epoch, threshold, start_value=0, end_value=1):
    res = end_value
    if epoch < threshold:
        res = start_value
    return res

# Custom rescale function for LPIPS normalization
def rescale_to_lpips_range(tensor, mean, std):
    # Rescale to [0, 1] range by normalizing using the mean and std
    rescaled_tensor = (tensor - torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)) / torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    # Rescale to [-1, 1] range
    rescaled_tensor = rescaled_tensor * 2.0 - 1.0
    return rescaled_tensor

def training_loop(n_epochs, optimizer, disc_optimizer, model, perceptual_loss,
                  device, data_loader, valid_loader, reset=False, p_loss_factor=1.0,
                  rec_loss_factor=1.0, max_grad_norm=1.0, epoch_start=0, save_img=True, 
                  show_img=False, usage_threshold=1.0):

    model.train()

    for epoch in range(epoch_start, n_epochs):
        loss_train = 0.0
        vq_loss_train = 0.0
        loss_gan = 0.0
        p_rec_loss_value = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit='batch')
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            imgs = imgs.to(device)

            outputs, vq_loss = model(imgs)
            
            gan_loss, f_out = model.calculate_discriminator_loss(imgs, outputs)
            
            # Rescale the output images to [-1, 1] range for perceptual loss
            rescaled_outputs = rescale_to_lpips_range(outputs, original_mean, original_std)
            rescaled_imgs = rescale_to_lpips_range(imgs, original_mean, original_std)
            
            p_loss = perceptual_loss(rescaled_outputs, rescaled_imgs)
            
            rec_loss = torch.abs(imgs - outputs)
            p_rec_loss = p_loss_factor * p_loss + rec_loss_factor * rec_loss
            
            p_rec_loss = p_rec_loss.mean()
            
            g_loss = -torch.mean(f_out)
            
            alpha = model.calculate_alpha(p_rec_loss, g_loss)
            
            gan_weight_value = gan_weight(epoch, 3)
            gan_loss *= gan_weight_value

            loss = p_rec_loss + vq_loss + (alpha * g_loss * gan_weight_value)

            loss_train += loss.item()
            vq_loss_train += vq_loss.item()
            loss_gan += gan_loss.item()
            p_rec_loss_value += p_rec_loss.item()
            
            optimizer.zero_grad()
            disc_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            gan_loss.backward()
            optimizer.step()
            disc_optimizer.step()

            progress_bar.set_postfix(loss=loss.item(), p_rec_loss=p_rec_loss.item(), gan_loss=gan_loss.item(), vq_loss=vq_loss.item())

        avg_loss_train = loss_train / len(data_loader)
        avg_vq_loss_train = vq_loss_train / len(data_loader)
        avg_gan_loss = loss_gan / len(data_loader)
        avg_p_rec_loss = p_rec_loss_value / len(data_loader)
       
        with open("f-waifu-vqvae_epoch-loss.txt", "a") as file:
            file.write(f"{avg_loss_train}\n")
        
        print('{} Epoch {}, Training loss {:.4f}, p_rec_loss {:.4f}, VQ loss {:.4f}, GAN Loss {:.4f}'.format(
            datetime.datetime.now(), epoch, avg_loss_train, avg_p_rec_loss, avg_vq_loss_train, avg_gan_loss))


        if epoch % 5 == 0:
            # Validation phase
            model.eval()
            loss_val = 0.0
            with torch.no_grad():
                for imgs, _ in valid_loader:
                    imgs = imgs.to(device)
                    outputs, vq_loss = model(imgs)
                    p_loss = perceptual_loss(rescale_to_lpips_range(outputs, original_mean, original_std), 
                                             rescale_to_lpips_range(imgs, original_mean, original_std))
                    rec_loss = torch.abs(imgs - outputs)
                    p_rec_loss = p_loss_factor * p_loss + rec_loss_factor * rec_loss
                    loss = p_rec_loss.mean() + vq_loss
                    loss_val += loss.item()

            avg_loss_val = loss_val / len(valid_loader)
            print(f'Val loss: {avg_loss_val}')
            with open("val-waifu-vqvae_epoch-loss.txt", "a") as file:
                file.write(f"{avg_loss_val}\n")


        if epoch % 5 == 0:
            if save_img:
                with torch.no_grad():
                    for valid_tensors, _ in valid_loader:
                        break

                    save_img_tensors_as_grid(valid_tensors, 8, "true1")
                    val_img, _ = model(valid_tensors.to(device))
                    save_img_tensors_as_grid(val_img, 8, "recon1")

            model_path = os.path.join('weights/', 'waifu-vqvae_epoch.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
        
        if reset:
            if epoch % 5 == 0 and previous_loss is not None and avg_loss_train > previous_loss * 1.25:
                print("reseting")
                with torch.no_grad():
                    for batch_imgs, _ in data_loader:
                        model.reset_underused_embeddings(batch_imgs.to(device), threshold=usage_threshold)
                        break

        previous_loss = avg_loss_train

if __name__ == "__main__":
    path = ''
    val_path = ''
    model_path = ''
    epoch = 0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model = VQVAE(latent_dim = 16, num_embeddings=512, beta=0.25, use_ema=False, e_width=64,d_width=64)
    model.to(device)
    
    optimizer = optim.AdamW(list(model.encoder.parameters())+
                            list(model.decoder.parameters())+
                            list(model.codebook.parameters()), lr=2e-4)
    disc_optimizer = optim.AdamW(model.discriminator.parameters(), lr=2e-4)

    perceptual_loss = lpips.LPIPS(net='vgg').eval()
    perceptual_loss.to(device)

    print(f"VQVAE param count: {count_parameters(model)}")
    print(f"Discriminator Param count: {count_parameters(model.discriminator)}")

    data_loader = get_data_loader(path, batch_size = 64, num_samples=20_000)
    val_loader = get_data_loader(val_path, batch_size = 64, num_samples=10_000)
    
    # Mean and std used for original normalization
    original_mean = (0.7002, 0.6099, 0.6036)
    original_std = (0.2195, 0.2234, 0.2097)
    
    '''
    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    '''
    
    with torch.no_grad():
        for valid_tensors, _ in val_loader:
            break

        save_img_tensors_as_grid(valid_tensors, 8, "true1")
        val_img, _ = model(valid_tensors.to(device))
        save_img_tensors_as_grid(val_img, 8, "recon1")

    training_loop(
        n_epochs=300,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        model=model,
        perceptual_loss=perceptual_loss,
        device=device,
        data_loader=data_loader,
        valid_loader=val_loader,
        epoch_start=epoch + 1,
    )
