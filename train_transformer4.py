import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
import os
from os.path import join
from models.build4 import build_model
from loss import IDMRFLoss
from models.Discriminator_ml import MsImageDis
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from dataset import CORP_REGION_SIZE, dataset_norm, dataset_two_corps
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
# this version is with normlized input with mean and std, all layers are normalized,
# change the order of the 'make_layer' with norm-activate-conv,and use the multi-scal D
# use two kind feature, horizon and vertical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer):
    gen.train()
    dis.train()

    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0

    for batch_idx, (gt, mask_img) in enumerate(train_loader):
        batchSize = mask_img.shape[0]
        gt, mask_img = gt.cuda(), mask_img.type(torch.FloatTensor).cuda()
        iner_img = gt[:, :, 32 : 32 + 128, 32 : 32 + 128]
        # I_groundtruth = torch.cat((I_l, I_r), 3)  # shape: B,C,H,W


        ## Generate Image
        I_pred, f_de = gen(mask_img)
        # I_pred = gen(mask_img)
        f_en = gen(iner_img, only_encode=True)

        print("gt shape:", gt.shape, "mask_img shape:", mask_img.shape, "iner_img shape:", iner_img.shape, "I_pred shape:", I_pred.shape, "f_de shape:", f_de.shape, "f_en shape:", f_en.shape)
        exit(1)
        # i_mask = torch.ones_like(gt)
        # i_mask[:, :, 32:32 + 128, 32:32 + 128] = 0
        # mask_pred = I_pred * i_mask
        mask_pred = I_pred[:, :, 32 : 32 + 128, 32 : 32 + 128]

        ## Compute losses
        ## Update Discriminator
        opt_dis.zero_grad()
        dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
        dis_loss = dis_adv_loss
        dis_loss.backward()
        opt_dis.step()

        # Pixel Reconstruction Loss
        pixel_rec_loss = mae(I_pred, gt) * 20

        # Texture Consistency Loss (IDMRF Loss)
        mrf_loss = (
            mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0)
            * 0.5
            / batchSize
        )
        # Feature Reconstruction Loss
        feat_rec_loss = mae(f_de, f_en.detach())

        # Update Generator
        gen_adv_loss = dis.calc_gen_loss(I_pred, gt)
        gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0)
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        acc_dis_adv_loss += dis_adv_loss.data

        if batch_idx % 10 == 0:
            print("train iter %d" % batch_idx)
            print("generate_loss:", gen_loss.item())
            print("dis_loss:", dis_loss.item())

    ## Tensor board
    writer.add_scalars(
        "train/generator_loss",
        {"Pixel Reconstruction Loss": acc_pixel_rec_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Texture Consistency Loss": acc_mrf_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Feature Reconstruction Loss": acc_feat_rec_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Adversarial Loss": acc_gen_adv_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/discriminator_loss",
        {"Adversarial Loss": acc_dis_adv_loss / len(train_loader.dataset)},
        epoch,
    )

# Training
def train_2(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer):
    gen.train()
    dis.train()

    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0

    for batch_idx, (gt, mask_img) in enumerate(train_loader):
        batchSize = mask_img.shape[0]
        gt, mask_img = gt.cuda(), mask_img.type(torch.FloatTensor).cuda()
        
        left_down_img = gt[:, :, 192 - CORP_REGION_SIZE :, 0 : CORP_REGION_SIZE]
        right_up_img = gt[:, :, 0 : CORP_REGION_SIZE, 192 - CORP_REGION_SIZE :]


        ## Generate Image
        I_pred, f_de = gen(mask_img)
        f_en_l = gen(left_down_img, only_encode=True)
        f_en_r = gen(right_up_img, only_encode=True)
        
        # TODO: maybe we can find a better way here to match the shape
        f_de = F.adaptive_avg_pool2d(f_de.permute(0, 3, 1, 2), output_size=(3, 3)).permute(0, 2, 3, 1)
        
        # print("gt shape:", gt.shape, "mask_img shape:", mask_img.shape, "left_down_img shape:", left_down_img.shape, "right_up_img shape:", right_up_img.shape, "I_pred shape:", I_pred.shape, "f_de shape:", f_de.shape, "f_en_l shape:", f_en_l.shape, "f_en_r shape:", f_en_r.shape)

        mask_pred_l = I_pred[:, :, 192 - CORP_REGION_SIZE :, 0 : CORP_REGION_SIZE]
        mask_pred_r = I_pred[:, :, 0 : CORP_REGION_SIZE, 192 - CORP_REGION_SIZE :]

        opt_dis.zero_grad()
        dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
        dis_loss = dis_adv_loss
        dis_loss.backward()
        opt_dis.step()

        # Pixel Reconstruction Loss
        pixel_rec_loss = mae(I_pred, gt) * 20

        # Texture Consistency Loss (IDMRF Loss)
        mrf_loss_l = (
            mrf((mask_pred_l.cuda(0) + 1) / 2.0, (left_down_img.cuda(0) + 1) / 2.0)
            * 0.5
            / batchSize
        )
        mrf_loss_r = (
            mrf((mask_pred_r.cuda(0) + 1) / 2.0, (right_up_img.cuda(0) + 1) / 2.0)
            * 0.5
            / batchSize
        )
        mrf_loss = mrf_loss_l + mrf_loss_r
        # Feature Reconstruction Loss
        feat_rec_loss_l = mae(f_de, f_en_l.detach())
        feat_rec_loss_r = mae(f_de, f_en_r.detach())
        feat_rec_loss = feat_rec_loss_l + feat_rec_loss_r
        

        # Update Generator
        gen_adv_loss = dis.calc_gen_loss(I_pred, gt)
        gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0)
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        acc_dis_adv_loss += dis_adv_loss.data

        if batch_idx % 10 == 0:
            print("train iter %d" % batch_idx)
            print("generate_loss:", gen_loss.item())
            print("dis_loss:", dis_loss.item())

    ## Tensor board
    writer.add_scalars(
        "train/generator_loss",
        {"Pixel Reconstruction Loss": acc_pixel_rec_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Texture Consistency Loss": acc_mrf_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Feature Reconstruction Loss": acc_feat_rec_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/generator_loss",
        {"Adversarial Loss": acc_gen_adv_loss / len(train_loader.dataset)},
        epoch,
    )
    writer.add_scalars(
        "train/discriminator_loss",
        {"Adversarial Loss": acc_dis_adv_loss / len(train_loader.dataset)},
        epoch,
    )


if __name__ == "__main__":
    SAVE_WEIGHT_DIR = "./checkpoints/former_resize_4-3/"
    SAVE_LOG_DIR = "./logs/logs_former_resize_4-3/"
    TRAIN_DATA_DIR = "./data-scenery-small"
    LOAD_WEIGHT_DIR = ""  # No pre-trained weights to load

    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--train_batch_size",
            type=int,
            help="batch size of training data",
            default=40,
        )
        parser.add_argument(
            "--test_batch_size", type=int, help="batch size of testing data", default=16
        )
        parser.add_argument("--epochs", type=int, help="number of epoches", default=300)
        parser.add_argument("--lr", type=float, help="learning rate", default=2e-4)
        parser.add_argument(
            "--alpha",
            type=float,
            help="learning rate decay for discriminator",
            default=0.1,
        )
        parser.add_argument(
            "--load_pretrain", type=bool, help="load pretrain weight", default=False
        )
        parser.add_argument(
            "--test_flag", type=bool, help="testing while training", default=False
        )
        parser.add_argument(
            "--adjoint", type=bool, help="if use adjoint in odenet", default=True
        )

        parser.add_argument(
            "--load_weight_dir",
            type=str,
            help="directory of pretrain model weights",
            default=LOAD_WEIGHT_DIR,
        )
        parser.add_argument(
            "--save_weight_dir",
            type=str,
            help="directory of saving model weights",
            default=SAVE_WEIGHT_DIR,
        )
        parser.add_argument(
            "--log_dir", type=str, help="directory of saving logs", default=SAVE_LOG_DIR
        )
        parser.add_argument(
            "--train_data_dir",
            type=str,
            help="directory of training data",
            default=TRAIN_DATA_DIR,
        )
        opts = parser.parse_args()
        return opts

    args = get_args()
    config = {}
    config["pre_step"] = 1
    config["TYPE"] = "swin"
    config["IMG_SIZE"] = 224
    config["SWIN.PATCH_SIZE"] = 4
    config["SWIN.IN_CHANS"] = 3
    config["SWIN.EMBED_DIM"] = 96
    config["SWIN.DEPTHS"] = [2, 2, 6, 2]
    config["SWIN.NUM_HEADS"] = [3, 6, 12, 24]
    config["SWIN.WINDOW_SIZE"] = 7
    config["SWIN.MLP_RATIO"] = 4.0
    config["SWIN.QKV_BIAS"] = True
    config["SWIN.QK_SCALE"] = None
    config["DROP_RATE"] = 0.0
    config["DROP_PATH_RATE"] = 0.2
    config["SWIN.PATCH_NORM"] = True
    config["TRAIN.USE_CHECKPOINT"] = False

    pred_step = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(
        join(args.log_dir, "SR_Stage_4%s" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    )

    # Initialize the model
    print("Initializing model...")
    gen = build_model(config).cuda()
    dis = MsImageDis().cuda()

    opt_gen = optim.Adam(
        gen.parameters(), lr=args.lr / 2, betas=(0, 0.9), weight_decay=1e-4
    )
    opt_dis = optim.Adam(
        dis.parameters(), lr=args.lr * 2, betas=(0, 0.9), weight_decay=1e-4
    )

    # Load pre-trained weight
    if args.load_pretrain:
        print("Loading model weight...at epoch 140")
        gen.load_state_dict(torch.load(join(args.load_weight_dir, "Gen_former_500")))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, "Dis_former_500")))

    # Load data
    print("Loading data...")
    transformations = transforms.Compose(
        [Resize(192), CenterCrop(192), ToTensor(), Normalize(mean, std)]
    )
    # train_data = dataset_norm(
    #     root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128
    # )
    train_data = dataset_two_corps(
        root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=CORP_REGION_SIZE
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    print("train data: %d images" % (len(train_loader.dataset)))

    # Train & test the model
    for epoch in range(1, args.epochs + 1):
        print("----Start training[%d]----" % epoch)
        # train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer)
        train_2(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer)

        # Save the model weight
        torch.save(
            gen.state_dict(), join(args.save_weight_dir, "Gen_former_%d" % epoch)
        )
        torch.save(
            dis.state_dict(), join(args.save_weight_dir, "Dis_former_%d" % epoch)
        )

    writer.close()
