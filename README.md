# DL Final Project

## GAN

### Train

```bash
python main.py --mode train
```

### Eval

```bash
python main.py --mode eval --checkpoint_path checkpoints/generator_epoch_20.pth --test_dir data-scenery-small-test
```

### TODO

- [x] Fix Dark image
- [x] Using mask to make sure the original image is not updated.
- [ ] Bigger image size
- [ ] Smaller crop size
- [ ] Generate only the missing part, instead of using mask (hard based on the GAN)
- [ ] Using two different images
