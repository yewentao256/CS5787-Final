import os
import numpy as np
from glob import glob

import torch
from torchvision import transforms
from PIL import Image

from new_train import UNetGenerator, crop_image

def preprocess_image(img, target_size=256):
    """
    对图像进行中心裁剪或调整大小，以确保图像尺寸为 target_size x target_size。
    """
    h, w, _ = img.shape
    if h > target_size and w > target_size:
        # 中心裁剪
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        img = img[top:top+target_size, left:left+target_size]
    else:
        # 调整大小
        img = np.array(Image.fromarray(img).resize((target_size, target_size), Image.BILINEAR))
    return img

def evaluate(img_path, generator, device, transform, target_size=256):
    generator.eval()
    try:
        # 使用 PIL 打开图像并转换为 RGB
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Warning: Unable to open image {img_path}. Error: {e}")
        # 创建一个空白图像作为占位符
        img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # 将 PIL 图像转换为 NumPy 数组
    img = np.array(img)
    
    # 对图像进行预处理（中心裁剪或调整大小到 target_size）
    preprocessed_img = preprocess_image(img, target_size)
    
    # 将预处理后的原始图像转换回 PIL 格式
    original_img_pil = Image.fromarray(preprocessed_img)
    
    # 使用 crop_image 函数获取带掩码的输入图像
    input_img, _ = crop_image(preprocessed_img, target_size)
    
    # 将输入图像转换为 PIL 格式
    input_img_pil = Image.fromarray(input_img)
    
    # 将输入图像转换为张量并移动到设备
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        gen_img = generator(input_tensor)
    
    # 合并输入和生成的部分
    output_tensor = input_tensor + gen_img
    output_tensor = torch.clamp(output_tensor, -1, 1)  # 避免超出范围
    
    # 将生成的张量转换为 NumPy 数组
    output_img = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # 将生成的图像从 [-1, 1] 映射到 [0, 255]
    output_img = ((output_img + 1) / 2 * 255).astype(np.uint8)

    
    return original_img_pil, input_img_pil, output_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化生成器并加载预训练权重
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load('checkpoints/generator_epoch_1.pth', map_location=device))
    generator.eval()
    
    # 测试图像路径
    test_image_paths = glob(os.path.join('data-scenery-small-test', '*'))
    
    # 创建主输出目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    for img_path in test_image_paths:
        # 生成原始图像、输入图像和生成结果
        original_img_pil, input_img_pil, output_img = evaluate(img_path, generator, device, transform, target_size=256)
        
        # 获取图像名称（不包含扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 为当前图像创建一个独立的文件夹
        img_output_dir = os.path.join(results_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 保存原始图像
        original_output_path = os.path.join(img_output_dir, "original.jpg")
        original_img_pil.save(original_output_path)
        
        # 保存输入图像
        input_output_path = os.path.join(img_output_dir, "input.jpg")
        input_img_pil.save(input_output_path)
        
        # 保存生成的图像
        generated_img_pil = Image.fromarray(output_img)
        generated_output_path = os.path.join(img_output_dir, "generated.jpg")
        generated_img_pil.save(generated_output_path)
        
        print(f"Saved original, input, and generated images to {img_output_dir}")

if __name__ == "__main__":
    main()
