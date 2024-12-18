# DL Final Project

## GAN

### Train

```bash
python main.py --mode train --epochs 30
python diffusion.py --mode train --epochs 30
```

### Train From Checkpoints

```bash
python main.py --mode train --checkpoint_path checkpoints/checkpoint_epoch_30.pth --epochs 50
```

### Eval

```bash
python main.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_95.pth --test_dir data-scenery-small-test
python diffusion.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_1.pth --test_dir data-scenery-small-test
```

### Eval with Two Images

Note: You can directly input any image, our code will help you to crop

```bash
python main.py --mode eval_2 --checkpoint_path checkpoints/checkpoint_epoch_20.pth --image2 data-scenery-small-test/istock-612x612.jpg --image1 data-scenery-small-test/pexels-pripicart.jpg
```

### TODO

- [x] Fix Dark image
- [x] Using mask to make sure the original image is not updated.
- [x] Adaptable crop size
- [x] Adding metrics
- [x] Update loss to optimize metrics(SSIM)
- [x] Using two different images
- [x] Update loss to optimize metrics(Perceptual loss-VGG 19)
