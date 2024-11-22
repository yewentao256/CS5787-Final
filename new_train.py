import os
import cv2
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# 超参数
num_epochs = 100
batch_size = 16
learning_rate = 2e-4
lambda_recon = 100

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# crop image to target size
def preprocess_image(img, target_size=256):
    h, w, _ = img.shape
    if h > target_size and w > target_size:
        # crop center
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        img = img[top:top+target_size, left:left+target_size]
    else:
        # 调整大小
        img = cv2.resize(img, (target_size, target_size))
    return img

def crop_image(img, target_size=256):
    # process image to target_size x target_size
    img = preprocess_image(img, target_size)

    left_bottom = img[target_size//2:, :target_size//2]
    right_top = img[:target_size//2, target_size//2:]

    # 创建输入图像，只包含截取的部分
    input_img = np.zeros_like(img)
    input_img[target_size//2:, :target_size//2] = left_bottom
    input_img[:target_size//2, target_size//2:] = right_top

    # 目标图像，未被截取的部分
    target_img = img.copy()
    target_img[target_size//2:, :target_size//2] = 0
    target_img[:target_size//2, target_size//2:] = 0

    return input_img, target_img


class SceneryDataset(Dataset):
    def __init__(self, image_paths, transform=None, target_size=256):
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
        except Exception as e:
            print(f"Warning: Unable to read image at path: {img_path}. Error: {e}. Returning a blank image.")
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        
        input_img, target_img = crop_image(img, self.target_size)
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


# UnetGenerator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # encoder
        self.encoder1 = self.contract_block(in_channels, features, 4, 2, 1)
        self.encoder2 = self.contract_block(features, features * 2, 4, 2, 1)
        self.encoder3 = self.contract_block(features * 2, features * 4, 4, 2, 1)
        self.encoder4 = self.contract_block(features * 4, features * 8, 4, 2, 1)
        self.encoder5 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder6 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder7 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder8 = self.contract_block(features * 8, features * 8, 4, 2, 1)

        # decoder
        self.decoder1 = self.expand_block(features * 8, features * 8, 4, 2, 1)
        self.decoder2 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder3 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder4 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder5 = self.expand_block(features * 16, features * 4, 4, 2, 1)
        self.decoder6 = self.expand_block(features * 8, features * 2, 4, 2, 1)
        self.decoder7 = self.expand_block(features * 4, features, 4, 2, 1)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # encode
        enc1 = self.encoder1(x)  # 128
        enc2 = self.encoder2(enc1)  # 64
        enc3 = self.encoder3(enc2)  # 32
        enc4 = self.encoder4(enc3)  # 16
        enc5 = self.encoder5(enc4)  # 8
        enc6 = self.encoder6(enc5)  # 4
        enc7 = self.encoder7(enc6)  # 2
        enc8 = self.encoder8(enc7)  # 1

        # decode
        dec1 = self.decoder1(enc8)  # 2
        dec1 = torch.cat([dec1, enc7], dim=1)
        dec2 = self.decoder2(dec1)  # 4
        dec2 = torch.cat([dec2, enc6], dim=1)
        dec3 = self.decoder3(dec2)  # 8
        dec3 = torch.cat([dec3, enc5], dim=1)
        dec4 = self.decoder4(dec3)  # 16
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec5 = self.decoder5(dec4)  # 32
        dec5 = torch.cat([dec5, enc3], dim=1)
        dec6 = self.decoder6(dec5)  # 64
        dec6 = torch.cat([dec6, enc2], dim=1)
        dec7 = self.decoder7(dec6)  # 128
        dec7 = torch.cat([dec7, enc1], dim=1)
        dec8 = self.decoder8(dec7)  # 256

        return dec8

# PatchDiscriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # input: (in_channels * 2) x 256 x 256
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),  # 64 x 128 x 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 64 x 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 32 x 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # 512 x 31 x 31
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # 1 x 30 x 30
        )

    def forward(self, x, y):
        # x: input image, y: generated or target image
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)

if __name__ == '__main__':
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_recon = nn.L1Loss()

    train_image_paths = glob(os.path.join('data-scenery', '*'))
    test_image_paths = glob(os.path.join('data-scenery-small-test', '*'))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = SceneryDataset(train_image_paths, transform=transform, target_size=256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        for i, (input_img, target_img) in enumerate(train_loader):
            input_img = input_img.to(device)  # [B, 3, 256, 256]
            target_img = target_img.to(device)  # [B, 3, 256, 256]
            real_img = input_img + target_img  # real image [B, 3, 256, 256]

            discriminator.zero_grad()

            real_labels = torch.ones((input_img.size(0), 1, 30, 30)).to(device)  # PatchGAN输出尺寸为30x30
            fake_labels = torch.zeros((input_img.size(0), 1, 30, 30)).to(device)

            real_output = discriminator(input_img, real_img)
            d_loss_real = criterion_GAN(real_output, real_labels)

            gen_img = generator(input_img)
            fake_img = input_img + gen_img
            fake_output = discriminator(input_img, fake_img.detach())
            d_loss_fake = criterion_GAN(fake_output, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            generator.zero_grad()

            gen_img = generator(input_img)
            fake_img = input_img + gen_img
            fake_output = discriminator(input_img, fake_img)

            g_adv_loss = criterion_GAN(fake_output, real_labels)
            g_rec_loss = criterion_recon(gen_img, target_img)
            g_loss = g_adv_loss + lambda_recon * g_rec_loss
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                    f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')
