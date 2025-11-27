import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from natsort import natsorted
from torch import nn, Tensor
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from mmseg.apis import init_model
 
# ---------------------------- 配置字典 ----------------------------
CONFIG = {
    # 模型配置
    "model": {
        "config_path": r"HWNet",
        "checkpoint_path": r"/home-nv1/2024xlh/experiments/train/ablation/SYSU/SYSU-train-IPNet-CA_250826_101811/checkpoint/best_cd_model_gen.pth",
        "target_layers": [
            ["model.encoder1"],
            ["model.encoder2"],
            ["model.encoder3"],
            ["model.encoder4"],
        ],
        "preview_model": True,
        "like_vit": False,
    },
    
    # 数据配置
    "data": {
        "imageA_dir": r"/home/207lab/change_detection_datasets/SYSU-CD-256/test/A",
        "imageB_dir": r"/home/207lab/change_detection_datasets/SYSU-CD-256/test/B",
        "label_dir": r"/home/207lab/change_detection_datasets/SYSU-CD-256/test/label",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    
    # CAM配置
    "cam": {
        "method": "EigenCAM",  # 支持的: GradCAM, GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM
        "base_save_dir": r"/home-nv1/2024xlh/grad-CAM/SYSU-eigencam/CA",
        "vis_results": False,
        "save_heatmap": True,
        "save_cam": True,
    },
    
    # 运行时配置
    "runtime": {
        "device": "cuda:3" if torch.cuda.is_available() else "cpu",
    }
}

# 定义支持的GradCam方法
METHOD_MAP = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "eigencam": EigenCAM,
    "eigengradcam": EigenGradCAM,
    "layercam": LayerCAM,
}

# 定义支持的颜色映射
COLOR_MAPS = {
    'JET': cv2.COLORMAP_JET,
    'HOT': cv2.COLORMAP_HOT,
    'COOL': cv2.COLORMAP_COOL,
    'BONE': cv2.COLORMAP_BONE,
    'MAGMA': cv2.COLORMAP_MAGMA,
    'PLASMA': cv2.COLORMAP_PLASMA,
    'TURBO': cv2.COLORMAP_TURBO,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'CIVIDIS': cv2.COLORMAP_CIVIDIS
}


# ---------------------------- 核心类定义 ----------------------------

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x: Tensor) -> Tensor:
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        return self.model(x1, x2)

 
class SemanticSegmentationTarget:
    """CAM目标定义"""
    def __init__(self, category: int, mask: np.ndarray, device: torch.device):
        self.category = category
        self.mask = torch.from_numpy(mask).to(device)
        
    def __call__(self, model_output: Tensor) -> Tensor:
        """目标计算"""
        return (model_output[self.category, :, :] * self.mask).sum()
 
# ---------------------------- 工具函数 ----------------------------
def init_models(config: dict) -> nn.Module:
    
    if config["model"]["config_path"] == "HWNet":
        print("Using HWNet model initialization...")
        
    device = torch.device(config["runtime"]["device"])
    
    # 1. 实例化你的模型
    from models.HWNet_v2 import HWNet  # 你的模型类
    model = HWNet(mode='parallel')  # 传入你的模型参数

    # 2. 加载权重
    state_dict = torch.load(config["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(state_dict)

    # 3. 封装成 ModelWrapper（如果 forward 返回和 mmseg 不同也要改）
    return ModelWrapper(model).to(device).eval()
 
def get_image_paths(config: dict) -> List[tuple]:
    """获取图像路径列表"""
    img_files = natsorted(os.listdir(config["data"]["label_dir"]))
    return [
        (
            os.path.join(config["data"]["imageA_dir"], f),
            os.path.join(config["data"]["imageB_dir"], f)
        ) for f in img_files
    ]
 
def preprocess_image_pair(image_path_A: str, image_path_B: str, config: dict) -> tuple:
    """预处理图像对"""
    def _process_single(path: str) -> tuple:
        img = np.array(Image.open(path))
        rgb_img = np.float32(img) / 255
        tensor = preprocess_image(rgb_img, 
                                mean=config["data"]["mean"],
                                std=config["data"]["std"])
        return tensor.to(config["runtime"]["device"]), rgb_img
    
    tensor_A, rgb_A = _process_single(image_path_A)
    tensor_B, rgb_B = _process_single(image_path_B)
    return (
        torch.cat([tensor_A, tensor_B], 1),
        torch.cat([tensor_B, tensor_A], 1),
        rgb_A,
        rgb_B
    )
 
def get_target_layer(model: nn.Module, layer_path: str):
    """安全获取目标层"""
    current = model
    for part in layer_path.split('.'):
        if '[' in part and ']' in part:
            name, index = part.split('[')
            index = int(index[:-1])
            current = getattr(current, name)[index]
        else:
            current = getattr(current, part)
    return current
 
def create_save_dirs(config: dict) -> List[str]:
    """创建保存目录"""
    save_dirs = []
    for layer_group in config["model"]["target_layers"]:
        dir_name = "_".join([p.replace("[", "_").replace("]", "") for p in layer_group])
        save_dir = os.path.join(config["cam"]["base_save_dir"], dir_name)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)
    return save_dirs
 
import matplotlib
import matplotlib.cm as cm

import matplotlib
import numpy as np
import cv2

def apply_heatmap(grayscale_cam, color_map='JET', invert=False):
    """
    生成带颜色的热力图（支持自定义红-橙-黄-绿渐变，红色为主）
    """

    # ----------- Step 1: 归一化到 [0, 255] uint8 -----------
    grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
    min_val, max_val = np.min(grayscale_cam), np.max(grayscale_cam)
    if max_val - min_val < 1e-6:
        grayscale_cam = np.zeros_like(grayscale_cam, dtype=np.uint8)
    else:
        grayscale_cam = ((grayscale_cam - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # ----------- Step 2: 应用 colormap -----------
    if color_map.lower() == 'custom':
        colors = [
            (0.0, "darkred"),
            (0.3, "red"),
            (0.6, "orange"),
            (0.85, "yellow"),
            (1.0, "green"),
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
        colormap = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        colormap = colormap.reshape(256, 1, 3)
        heatmap = cv2.applyColorMap(grayscale_cam, colormap)
    else:
        cmap = COLOR_MAPS.get(color_map.upper(), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(grayscale_cam, cmap)

    # ----------- Step 3: 可选反色 -----------
    if invert:
        heatmap = cv2.bitwise_not(heatmap)

    return heatmap



# ---------------------------- 核心处理逻辑 ----------------------------
def generate_cam_results(
    model: nn.Module,
    input_tensor: Tensor,
    targets: List[SemanticSegmentationTarget],
    target_layers: List[nn.Module],
    config: dict
) -> np.ndarray:
    """生成CAM结果"""
    cam_class = METHOD_MAP[config["cam"]["method"].lower()]
    reshape_fn = reshape_transform if config["model"]["like_vit"] else None
    
    # 修复1：移除use_cuda参数
    with cam_class(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_fn
    ) as cam:
        # 修复2：添加显式设备管理
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        # 修复3：添加异常处理
        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            print(f"生成CAM时发生错误: {str(e)}")
            return np.zeros_like(input_tensor.cpu().numpy()[0, 0])
        
def safe_generate_heatmap(grayscale_cam: np.ndarray) -> np.ndarray:
    """安全生成热力图"""
    # 处理NaN和Inf
    grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 检查数值范围
    if np.all(grayscale_cam == 0):
        return np.zeros_like(grayscale_cam, dtype=np.uint8)
    
    # 安全归一化
    min_val = np.min(grayscale_cam)
    max_val = np.max(grayscale_cam)
    if max_val - min_val < 1e-6:  # 防止除以零
        return np.zeros_like(grayscale_cam, dtype=np.uint8)
    
    normalized = (grayscale_cam - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)
 
 
def process_single_image(
    model: nn.Module,
    config: dict,
    image_data: tuple,
    img_paths: tuple,
    save_dirs: List[str]
):
    """处理单个图像对（改进版）"""
    # 解包数据
    input_tensor_AB, input_tensor_BA, rgb_A, rgb_B = image_data
    img_A_path, img_B_path = img_paths
    base_name = os.path.basename(img_A_path).split('.')[0]
 
    # 定义输入组合及其对应的元数据
    input_combinations = [
        {
            "tensor": input_tensor_AB,
            "direction": "AB",
            "base_image": rgb_A,
            "ref_image": rgb_B
        },
        {
            "tensor": input_tensor_BA,
            "direction": "BA",
            "base_image": rgb_B,
            "ref_image": rgb_A
        }
    ]
 
    # 对每个输入组合进行处理
    for combo in input_combinations:
        # 模型推理
        with torch.no_grad():
            output = model(combo["tensor"])
 
        if output.shape[1] == 1:
            # 单通道：直接取 sigmoid 概率
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = np.float32(pred_mask > 0.3)  # 阈值可调
            category_idx = 0
        else:
            # 二分类及以上
            pred_mask = torch.softmax(output, 1).cpu()
            pred_mask = pred_mask.argmax(1).numpy()[0]
            # 假设变化类是索引 1（no_change=0, change=1）
            category_idx = 1
            mask = np.float32(pred_mask == category_idx)

        # 创建目标
        targets = [SemanticSegmentationTarget(
            category_idx,
            mask,
            torch.device(config["runtime"]["device"])
        )] 
    
        # 处理每个目标层组
        for layer_group, save_dir in zip(config["model"]["target_layers"], save_dirs):
            # 获取目标层
            target_layers = [
                get_target_layer(model, layer_path)
                for layer_path in layer_group
            ]
            
            # 生成CAM
            grayscale_cam = generate_cam_results(
                model=model,
                input_tensor=combo["tensor"],
                targets=targets,
                target_layers=target_layers,
                config=config
            )
 
            # 对两个图像进行叠加
            for img_type in ["base", "ref"]:
                # 选择要叠加的图像
                target_img = combo["base_image"] if img_type == "base" else combo["ref_image"]
                suffix = f"{combo['direction']}_{img_type}"
 
                # 生成叠加图像
                # cam_image = show_cam_on_image(target_img, grayscale_cam, colormap=cv2.COLORMAP_JET, use_rgb=True)
                
                heatmap_color = apply_heatmap(grayscale_cam, color_map='JET')
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                # 再叠加到原图上
                cam_image = (0.5 * target_img * 255 + 0.5 * heatmap_color).astype(np.uint8)
                
                
                # 保存结果
                if config["cam"]["save_cam"]:
                    Image.fromarray(cam_image).save(
                        os.path.join(save_dir, f"{base_name}_{suffix}.png"))
                
                if config["cam"]["save_heatmap"]:
                    
                    grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0, posinf=1.0, neginf=0.0)
                    min_val = np.min(grayscale_cam)
                    max_val = np.max(grayscale_cam)
                    denominator = max_val - min_val
                    if denominator < 1e-6:
                        heatmap = np.zeros_like(grayscale_cam, dtype=np.uint8)
                    else:
                        heatmap = ((grayscale_cam - min_val) / denominator * 255).astype(np.uint8)
                    
                    # heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    # 生成 Grad-CAM 热力图
                    heatmap_color = apply_heatmap(grayscale_cam, color_map='JET')  # 用 MAGMA 风格
                    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    
                    
                    Image.fromarray(heatmap_rgb).save(
                        os.path.join(save_dir, f"{base_name}_{suffix}_heatmap.png"))
# ---------------------------- ViT支持 ----------------------------
def reshape_transform(tensor: Tensor) -> Tensor:
    """ViT类模型的特征变换"""
    return tensor.reshape(
        tensor.size(0),
        int(np.sqrt(tensor.size(1))),
        int(np.sqrt(tensor.size(1))),
        tensor.size(2),
    ).permute(0, 3, 1, 2)
 
# ---------------------------- 主函数 ----------------------------
def main():
    # 初始化模型
    print("Initializing model...")
    model = init_models(CONFIG)
    
    # 创建保存目录
    print("Creating save directories...")
    save_dirs = create_save_dirs(CONFIG)
    
    # 获取图像路径
    print("Loading image paths...")
    image_paths = get_image_paths(CONFIG)
    
    # 处理每个图像对
    print("Processing images...")
    try:
        for img_paths in tqdm(image_paths, desc="Processing"):
            img_A, img_B = img_paths
            try:
                image_data = preprocess_image_pair(img_A, img_B, CONFIG)
                process_single_image(
                    model=model,
                    config=CONFIG,
                    image_data=image_data,
                    img_paths=(img_A, img_B),
                    save_dirs=save_dirs
                )
            except Exception as e:
                print(f"处理 {img_A} 时发生错误: {str(e)}")
                continue
    finally:
        # 修复5：显式清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
if __name__ == "__main__":
    main()