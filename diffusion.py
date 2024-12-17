import os
from glob import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
import torch.nn.functional as F

BATCH_SIZE = 32
LEARNING_RATE = 2e-4
TARGET_SIZE = 256
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
CROP_RATIO = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_image(img, crop_ratio=CROP_RATIO):
    assert 0 < crop_ratio < 1
    _, H, W = img.shape
    H_crop = int(H * crop_ratio)
    W_crop = int(W * crop_ratio)

    # target_img 保持完整图像
    target_img = img.clone()

    # input_img 在未知区域用0填充
    input_img = img.clone()
    # 未知区域为图像的左下角和右上角两个方块
    input_img[:, H - H_crop :, :W_crop] = 0
    input_img[:, :H_crop, W - W_crop :] = 0

    # mask：1表示已知区域，0表示未知区域
    mask = torch.ones((1, H, W), dtype=img.dtype, device=img.device)
    mask[:, H - H_crop :, :W_crop] = 0
    mask[:, :H_crop, W - W_crop :] = 0

    return input_img, target_img, mask


def combine_two_images_into_input(
    img1, img2, target_size=TARGET_SIZE, crop_ratio=CROP_RATIO
):
    H_crop, W_crop = int(target_size * crop_ratio), int(target_size * crop_ratio)
    img1_resized, _, mask = crop_image(img1)
    img2_resized, _, _ = crop_image(img2)

    input_img = torch.zeros(3, target_size, target_size)
    input_img[:, target_size - H_crop :, :W_crop] = img1_resized[:, target_size - H_crop :, :W_crop]
    input_img[:, :H_crop, target_size - W_crop :] = img2_resized[:, :H_crop, target_size - W_crop :]

    return input_img, mask

transform = transforms.Compose(
    [
        transforms.Resize(TARGET_SIZE),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

class SceneryDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise Exception(f"Unable to open image {img_path}. Error: {e}")

        if self.transform:
            img = self.transform(img)

        return crop_image(img)

def get_timestep_embedding(timesteps, dim):
    # 基于高频正余弦函数的时间步编码
    # timesteps: [B], int
    # 返回 [B, dim] 的编码
    half_dim = dim // 2
    freq = torch.exp(
        torch.log(torch.tensor(10000.0, device=timesteps.device)) * 
        torch.linspace(0, 1, half_dim, device=timesteps.device)
    )
    freq = freq.unsqueeze(0)  # [1, half_dim]
    # timesteps为整数，将其转换为float
    angles = timesteps.float().unsqueeze(1) * freq  # [B, half_dim]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim]
    if dim % 2 == 1:  # 如果为奇数则再补一列
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(time_embed_dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        x = self.linear1(t)
        x = self.act(x)
        x = self.linear2(x)
        return x
class UNetDiffusionModel(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, features=64, time_embed_dim=128):
        super(UNetDiffusionModel, self).__init__()
        self.time_embed_dim = time_embed_dim

        # 时间嵌入 MLP
        self.time_mlp = TimeEmbedding(time_embed_dim, hidden_dim=features*8)

        # 编码器
        self.encoder1 = self.encode_block(in_channels, features, 4, 2, 1)
        self.encoder2 = self.encode_block(features, features * 2, 4, 2, 1)
        self.encoder3 = self.encode_block(features * 2, features * 4, 4, 2, 1)
        self.encoder4 = self.encode_block(features * 4, features * 8, 4, 2, 1)
        self.encoder5 = self.encode_block(features * 8, features * 8, 4, 2, 1)
        self.encoder6 = self.encode_block(features * 8, features * 8, 4, 2, 1)
        self.encoder7 = self.encode_block(features * 8, features * 8, 4, 2, 1)
        self.encoder8 = self.encode_block(features * 8, features * 8, 4, 2, 1)

        # 解码器
        self.decoder1 = self.decode_block(features * 8, features * 8, 4, 2, 1)
        self.decoder2 = self.decode_block(features * 16, features * 8, 4, 2, 1)
        self.decoder3 = self.decode_block(features * 16, features * 8, 4, 2, 1)
        self.decoder4 = self.decode_block(features * 16, features * 8, 4, 2, 1)
        self.decoder5 = self.decode_block(features * 16, features * 4, 4, 2, 1)
        self.decoder6 = self.decode_block(features * 8, features * 2, 4, 2, 1)
        self.decoder7 = self.decode_block(features * 4, features, 4, 2, 1)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )

    def encode_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def decode_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        # 时间步嵌入
        t_embed = get_timestep_embedding(t, self.time_embed_dim)  # [B, time_embed_dim]
        t_embed = self.time_mlp(t_embed)  # [B, features*8]
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)  # [B, features*8, 1, 1]

        # 编码
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        enc8 = self.encoder8(enc7)

        # 在bottleneck处添加时间嵌入
        enc8 = enc8 + t_embed

        # 解码（不重复添加t_embed，以避免通道数不匹配）
        dec1 = self.decoder1(enc8)
        dec1 = torch.cat([dec1, enc7], dim=1)

        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc6], dim=1)

        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc5], dim=1)

        dec4 = self.decoder4(dec3)
        dec4 = torch.cat([dec4, enc4], dim=1)

        dec5 = self.decoder5(dec4)
        dec5 = torch.cat([dec5, enc3], dim=1)

        dec6 = self.decoder6(dec5)
        dec6 = torch.cat([dec6, enc2], dim=1)

        dec7 = self.decoder7(dec6)
        dec7 = torch.cat([dec7, enc1], dim=1)

        dec8 = self.decoder8(dec7)
        return dec8


def linear_beta_schedule(timesteps, start=1e-4, end=2e-2):
    return torch.linspace(start, end, timesteps)

T = 1000
betas = linear_beta_schedule(T).to(device)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)
alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alpha_cumprod[:-1]], dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = sqrt_alpha_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

def p_sample(model, x, t, input_img, mask):
    # x: 当前带噪声的图像
    # 条件输入合成: noisy_image(=x), mask, input_img
    # 模型输入通道=7
    cond_input = torch.cat([x, mask, input_img], dim=1)
    noise_pred = model(cond_input, t)
    x0_pred = (x - sqrt_one_minus_alpha_cumprod[t].reshape(-1,1,1,1)*noise_pred) / sqrt_alpha_cumprod[t].reshape(-1,1,1,1)

    if t > 0:
        noise = torch.randn_like(x)
        alpha_t_prev = alpha_cumprod_prev[t]
        x_prev = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev)*noise
    else:
        x_prev = x0_pred

    return x_prev

def p_sample_loop(model, shape, input_img, mask):
    x = torch.randn(shape, device=device)
    x = input_img * mask + x * (1 - mask)  # 初始化

    for i in reversed(range(T)):
        t = torch.tensor([i], device=device).long().expand(shape[0])
        x = p_sample(model, x, t, input_img, mask)
        # 每一步采样后，重置已知区域保持不变
        x = input_img * mask + x * (1 - mask)
        
    return x


def train(args):
    num_epochs = args.epochs
    checkpoint_path = args.checkpoint_path

    model = UNetDiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    start_epoch = 0
    train_image_paths = glob(os.path.join(args.train_dir, "*"))
    if not train_image_paths:
        print(f"No training images found in {args.train_dir}. Exiting training.")
        return

    train_dataset = SceneryDataset(train_image_paths, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Starting Training on {device}... Parameters: {args}")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (input_img, target_img, mask) in enumerate(train_loader):
            input_img, target_img, mask = input_img.to(device), target_img.to(device), mask.to(device)

            optimizer.zero_grad()
            
            t = torch.randint(0, T, (input_img.size(0),), device=device).long()
            noise = torch.randn_like(target_img)
            x_t = q_sample(target_img, t, noise=noise)

            # 合成noisy_image
            noisy_image = input_img * mask + x_t * (1 - mask)
            model_input = torch.cat([noisy_image, mask, input_img], dim=1)

            noise_pred = model(model_input, t)
            unknown_area = (1 - mask)  # 未知区域
            loss = F.mse_loss(noise_pred * unknown_area, noise * unknown_area)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % args.log_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed. Avg Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"
        )
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoints for epoch {epoch+1} at {checkpoint_path}.")

    print("Training Completed.")


def evaluate(args):
    checkpoint_path = args.checkpoint_path
    test_dir = args.test_dir

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist. Exiting evaluation.")
        return

    model = UNetDiffusionModel().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    )
    model.eval()

    test_image_paths = glob(os.path.join(test_dir, "*"))
    if not test_image_paths:
        print(f"No test images found in {test_dir}. Exiting evaluation.")
        return

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    psnr_values = []
    ssim_values = []

    print("Starting Evaluation...")
    for img_path in test_image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Unable to open image {img_path}. Error: {e}")
            continue
        img_transformed = transform(img)
        input_img, target_img, mask = crop_image(img_transformed)
        input_tensor = input_img.unsqueeze(0).to(device)
        mask_tensor = mask.unsqueeze(0).to(device)
        original_tensor = img_transformed.unsqueeze(0).to(device)

        with torch.no_grad():
            shape = original_tensor.shape
            sampled = p_sample_loop(model, shape, input_tensor, mask_tensor)

        output_tensor = torch.clamp(sampled, -1, 1)
        output_tensor_scaled = output_tensor * 0.5 + 0.5
        original_tensor_scaled = original_tensor * 0.5 + 0.5

        psnr_value = peak_signal_noise_ratio(
            output_tensor_scaled, original_tensor_scaled, data_range=1.0
        ).item()
        ssim_value = structural_similarity_index_measure(
            output_tensor_scaled, original_tensor_scaled, data_range=1.0
        ).item()

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        # 更新FID度量
        output_uint8 = (output_tensor_scaled * 255).to(torch.uint8)
        target_uint8 = (original_tensor_scaled * 255).to(torch.uint8)
        fid_metric.update(output_uint8, real=False)
        fid_metric.update(target_uint8, real=True)

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_output_dir = os.path.join(RESULTS_DIR, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        original_output_path = os.path.join(img_output_dir, "original.jpg")
        input_output_path = os.path.join(img_output_dir, "input.jpg")
        generated_output_path = os.path.join(img_output_dir, "generated.jpg")
        save_image(original_tensor_scaled.cpu(), original_output_path)
        save_image((input_tensor * 0.5 + 0.5).cpu(), input_output_path)
        save_image(output_tensor_scaled.cpu(), generated_output_path)
        print(f"Saved original, input, and generated images to {img_output_dir}")

    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    fid_score = fid_metric.compute().item()

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"FID Score: {fid_score:.4f}")


def evaluate_two_images(args):
    checkpoint_path = args.checkpoint_path
    image1_path = args.image1
    image2_path = args.image2

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist. Exiting evaluation.")
        return

    if not os.path.exists(image1_path):
        print(f"Image file {image1_path} does not exist. Exiting evaluation.")
        return

    if not os.path.exists(image2_path):
        print(f"Image file {image2_path} does not exist. Exiting evaluation.")
        return

    model = UNetDiffusionModel().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)["model_state_dict"]
    )
    model.eval()

    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
    except Exception as e:
        print(f"Unable to open image(s). Error: {e}")
        return

    img1_transformed = transform(img1)
    img2_transformed = transform(img2)

    input_img, mask = combine_two_images_into_input(img1_transformed, img2_transformed)
    input_tensor = input_img.unsqueeze(0).to(device)
    mask_tensor = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        shape = input_tensor.shape
        sampled = p_sample_loop(model, shape, input_tensor, mask_tensor)

    output_tensor = torch.clamp(sampled, -1, 1)
    output_tensor_scaled = output_tensor * 0.5 + 0.5
    input_tensor_scaled = input_tensor * 0.5 + 0.5
    img1_transformed = img1_transformed * 0.5 + 0.5
    img2_transformed = img2_transformed * 0.5 + 0.5

    img1_name = os.path.splitext(os.path.basename(image1_path))[0]
    img2_name = os.path.splitext(os.path.basename(image2_path))[0]
    combined_output_dir = os.path.join(RESULTS_DIR, f"{img1_name}_{img2_name}_eval_2")
    os.makedirs(combined_output_dir, exist_ok=True)

    original_1_path = os.path.join(combined_output_dir, "image1.jpg")
    original_2_path = os.path.join(combined_output_dir, "image2.jpg")
    input_output_path = os.path.join(combined_output_dir, "input_combined.jpg")
    generated_output_path = os.path.join(combined_output_dir, "generated_combined.jpg")
    save_image(img1_transformed.cpu(), original_1_path)
    save_image(img2_transformed.cpu(), original_2_path)
    save_image(input_tensor_scaled.cpu(), input_output_path)
    save_image(output_tensor_scaled.cpu(), generated_output_path)
    print(f"Processed and saved combined images to {combined_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "eval_2"], default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_dir", type=str, default="data-scenery")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoints/checkpoint_epoch_20.pth"
    )
    parser.add_argument("--test_dir", type=str, default="data-scenery-small-test")
    parser.add_argument(
        "--image1", type=str, help="Path to first image for eval_2 mode"
    )
    parser.add_argument(
        "--image2", type=str, help="Path to second image for eval_2 mode"
    )

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "eval_2":
        evaluate_two_images(args)
