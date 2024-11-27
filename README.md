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
- [x] Adaptable crop size
- [x] Adding metrics
- [ ] Update loss to optimize metrics(doing)
- [ ] Update Model Structure to optimize loss.
- [ ] Bigger image size
- [ ] Using two different images
- [ ] Generate only the missing part, instead of using mask (hard based on GAN)
- [ ] Random crop region (hard based on GAN)
