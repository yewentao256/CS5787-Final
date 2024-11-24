import os
import numpy as np
from glob import glob
import argparse

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DEFAULT_NUM_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LAMBDA_RECON = 100
DEFAULT_TARGET_SIZE = 256
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"


os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_image(img):
    _, H, W = img.shape
    left_bottom = img[:, H // 2 :, : W // 2]  # [C, H/2, W/2]
    right_top = img[:, : H // 2, W // 2 :]  # [C, H/2, W/2]

    # input: left bottom and right top
    input_img = torch.full_like(img, -1)
    input_img[:, H // 2 :, : W // 2] = left_bottom
    input_img[:, : H // 2, W // 2 :] = right_top

    # target: right bottom and left top
    target_img = img.clone()
    target_img[:, H // 2 :, : W // 2] = -1
    target_img[:, : H // 2, W // 2 :] = -1
    return input_img, target_img


class ConditionalCenterCropOrResize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        if h > self.target_size and w > self.target_size:
            # Center crop
            img = transforms.functional.center_crop(img, self.target_size) # type: ignore
        else:
            # Resize
            img = transforms.functional.resize( # type: ignore
                img, (self.target_size, self.target_size), interpolation=Image.BILINEAR
            )
        return img


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

        input_img, target_img = crop_image(img)
        return input_img, target_img


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.encoder1 = self.contract_block(in_channels, features, 4, 2, 1)
        self.encoder2 = self.contract_block(features, features * 2, 4, 2, 1)
        self.encoder3 = self.contract_block(features * 2, features * 4, 4, 2, 1)
        self.encoder4 = self.contract_block(features * 4, features * 8, 4, 2, 1)
        self.encoder5 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder6 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder7 = self.contract_block(features * 8, features * 8, 4, 2, 1)
        self.encoder8 = self.contract_block(features * 8, features * 8, 4, 2, 1)

        # Decoder
        self.decoder1 = self.expand_block(features * 8, features * 8, 4, 2, 1)
        self.decoder2 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder3 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder4 = self.expand_block(features * 16, features * 8, 4, 2, 1)
        self.decoder5 = self.expand_block(features * 16, features * 4, 4, 2, 1)
        self.decoder6 = self.expand_block(features * 8, features * 2, 4, 2, 1)
        self.decoder7 = self.expand_block(features * 4, features, 4, 2, 1)
        self.decoder8 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encode
        enc1 = self.encoder1(x)  # 128
        enc2 = self.encoder2(enc1)  # 64
        enc3 = self.encoder3(enc2)  # 32
        enc4 = self.encoder4(enc3)  # 16
        enc5 = self.encoder5(enc4)  # 8
        enc6 = self.encoder6(enc5)  # 4
        enc7 = self.encoder7(enc6)  # 2
        enc8 = self.encoder8(enc7)  # 1

        # Decode
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
            # Input: (in_channels * 2) x 256 x 256
            nn.Conv2d(
                in_channels * 2, 64, kernel_size=4, stride=2, padding=1
            ),  # 64 x 128 x 128
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


def train(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lambda_recon = args.lambda_recon
    target_size = args.target_size

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_recon = nn.L1Loss()

    train_image_paths = glob(os.path.join(args.train_dir, "*"))
    if not train_image_paths:
        print(f"No training images found in {args.train_dir}. Exiting training.")
        return

    transform = transforms.Compose(
        [
            ConditionalCenterCropOrResize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_dataset = SceneryDataset(train_image_paths, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print("Starting Training...")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for i, (input_img, target_img) in enumerate(train_loader):
            input_img = input_img.to(device)  # [B, 3, 256, 256]
            target_img = target_img.to(device)  # [B, 3, 256, 256]
            real_img = input_img + target_img  # real image [B, 3, 256, 256]

            # Train Discriminator
            discriminator.zero_grad()

            real_labels = torch.ones((input_img.size(0), 1, 30, 30)).to(device)
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

            # Train Generator
            generator.zero_grad()

            gen_img = generator(input_img)
            fake_img = input_img + gen_img
            fake_output = discriminator(input_img, fake_img)

            g_adv_loss = criterion_GAN(fake_output, real_labels)
            g_rec_loss = criterion_recon(gen_img, target_img)
            g_loss = g_adv_loss + lambda_recon * g_rec_loss
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            if (i + 1) % args.log_interval == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                    f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}"
                )

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Completed. Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}"
        )

        # Save model checkpoints
        torch.save(
            generator.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth"),
        )
        torch.save(
            discriminator.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch+1}.pth"),
        )
        print(f"Saved checkpoints for epoch {epoch+1}.")

    print("Training Completed.")


def evaluate(args):
    checkpoint_path = args.checkpoint_path
    test_dir = args.test_dir
    results_dir = args.results_dir
    target_size = args.target_size

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} does not exist. Exiting evaluation.")
        return

    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    test_image_paths = glob(os.path.join(test_dir, "*"))
    if not test_image_paths:
        print(f"No test images found in {test_dir}. Exiting evaluation.")
        return

    transform = transforms.Compose(
        [
            ConditionalCenterCropOrResize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    def evaluate_image(img_path, generator, device, transform):
        generator.eval()
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise Exception(f"Unable to open image {img_path}. Error: {e}")

        img_transformed = transform(img)
        input_img, _ = crop_image(img_transformed)
        input_tensor = input_img.unsqueeze(0).to(device)

        with torch.no_grad():
            gen_img = generator(input_tensor)

        output_tensor = input_tensor + gen_img + 1
        output_tensor = torch.clamp(output_tensor, -1, 1)  # Avoid out of range

        # Convert tensors to images
        input_img_np = (input_img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        output_img_np = (
            output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        ) * 255

        input_img_pil = Image.fromarray(input_img_np.astype(np.uint8))
        output_img_pil = Image.fromarray(output_img_np.astype(np.uint8))

        # Original image (after resizing/cropping)
        original_img_pil = transforms.functional.to_pil_image( # type: ignore
            img_transformed * 0.5 + 0.5
        )

        return original_img_pil, input_img_pil, output_img_pil

    print("Starting Evaluation...")
    for img_path in test_image_paths:
        original_img_pil, input_img_pil, output_img_pil = evaluate_image(
            img_path, generator, device, transform
        )

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_output_dir = os.path.join(results_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Save image
        original_output_path = os.path.join(img_output_dir, "original.jpg")
        original_img_pil.save(original_output_path)
        input_output_path = os.path.join(img_output_dir, "input.jpg")
        input_img_pil.save(input_output_path)
        generated_output_path = os.path.join(img_output_dir, "generated.jpg")
        output_img_pil.save(generated_output_path)

        print(f"Saved original, input, and generated images to {img_output_dir}")

    print("Evaluation Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or Evaluate the UNet Generator Model."
    )
    subparsers = parser.add_subparsers(
        dest="mode", help="Mode of operation: train or eval", required=True
    )

    # Training sub-command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    train_parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizers",
    )
    train_parser.add_argument(
        "--lambda_recon",
        type=float,
        default=DEFAULT_LAMBDA_RECON,
        help="Weight for reconstruction loss",
    )
    train_parser.add_argument(
        "--train_dir",
        type=str,
        default="data-scenery",
        help="Directory containing training images",
    )
    train_parser.add_argument(
        "--target_size", type=int, default=DEFAULT_TARGET_SIZE, help="Target image size"
    )
    train_parser.add_argument(
        "--log_interval", type=int, default=10, help="How often to log training status"
    )

    # Evaluation sub-command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the generator checkpoint file",
    )
    eval_parser.add_argument(
        "--test_dir",
        type=str,
        default="data-scenery-small-test",
        help="Directory containing test images",
    )
    eval_parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory to save evaluation results",
    )
    eval_parser.add_argument(
        "--target_size", type=int, default=DEFAULT_TARGET_SIZE, help="Target image size"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        parser.print_help()
