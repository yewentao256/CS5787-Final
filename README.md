# DL Final Project

## GAN

### Train

```bash
python main.py --mode train
```

### Train From Checkpoints

```bash
python main.py --mode train --checkpoint_path checkpoints/checkpoint_epoch_1.pth
```

### Eval

```bash
python main.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_1.pth --test_dir data-scenery-small-test
```

### Eval with Two Images

Note: You can directly input any image, our code will help you to crop

```bash
python main.py --mode eval --checkpoint_path checkpoints/checkpoint_epoch_1.pth --image1 data-scenery-small-test/istock-612x612.jpg --image2 data-scenery-small-test/pexels-pripicart.jpg
python main.py --mode eval_2 --checkpoint_path checkpoints/checkpoint_epoch_1.pth --image1 data-scenery-small-test/lake-1280.jpg --image2 data-scenery-small-test/pexels-pripicart.jpg
python main.py --mode eval_2 --checkpoint_path checkpoints/checkpoint_epoch_1.pth --image1 data-scenery-small-test/lake-1280.jpg --image2 data-scenery-small-test/istock-612x612.jpg
python main.py --mode eval_2 --checkpoint_path checkpoints/checkpoint_epoch_1.pth --image1 data-scenery-small-test/lake-1280.jpg --image2 data-scenery-small-test/pexels-akos-szabo.jpg
```

### TODO

- [x] Fix Dark image
- [x] Using mask to make sure the original image is not updated.
- [x] Adaptable crop size
- [x] Adding metrics
- [x] Update loss to optimize metrics(SSIM)
- [x] Using two different images
- [x] Update loss to optimize metrics(Other)
- [ ] Update Model Structure to optimize loss.
- [ ] Bigger image size
- [ ] Generate only the missing part, instead of using mask (hard based on GAN)
- [ ] Random crop region (hard based on GAN)
