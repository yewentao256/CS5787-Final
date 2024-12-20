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
python diffusion.py --mode train --checkpoint_path checkpoints/checkpoint_epoch_21.pth --epochs 100
```

### Eval

```bash
python main.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_55.pth --test_dir data-scenery-small-test
python diffusion.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_21.pth --test_dir data-scenery-small-test
```

### Eval with Two Images

Note: You can directly input any image, our code will help you to crop

```bash
python main.py --mode eval_2 --checkpoint_path checkpoints/checkpoint_epoch_20.pth --image2 data-scenery-small-test/istock-612x612.jpg --image1 data-scenery-small-test/pexels-pripicart.jpg
```

## Diffusion-Non pretrained

### Train

```bash
python diffusion.py --mode train --epochs 30
```

### Train From Checkpoints

```bash
python diffusion.py --mode train --checkpoint_path checkpoints/checkpoint_epoch_21.pth --epochs 100
```

### Eval

```bash
python diffusion.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_21.pth --test_dir data-scenery-small-test
```

## Diffusion-Pretrained

Please see [ipynb](pretrained_diffusion.ipynb)
