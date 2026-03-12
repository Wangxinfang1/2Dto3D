import os
import cv2
import json
import numpy as np
from shapely.geometry import Polygon, box
from shapely.affinity import translate
from tqdm import tqdm
import math

# ================= 配置区域 =================
# 输入大图和JSON的文件夹
INPUT_DIR = "/home/wxf/dataset/train_raw" 
# 输出切片后的文件夹
OUTPUT_DIR = "/home/wxf/dataset/train" 

# 切片大小 (建议 1024 或 1280，配合 5090 显卡)
SLICE_SIZE = 1024
# 重叠率 (防止目标正好在边缘被切断，0.2 表示重叠 20%)
OVERLAP = 0.2 

# 过滤阈值：如果切出来的楼盘轮廓面积小于这个值，就忽略（防止边缘产生极小的噪点）
MIN_AREA_THRESHOLD = 50 
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def slice_single_image(image_path, json_path, output_img_dir, output_json_dir):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    img_h, img_w = img.shape[:2]
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    
    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    shapes = json_data.get('shapes', [])
    
    # 计算步长
    stride = int(SLICE_SIZE * (1 - OVERLAP))
    
    # 计算切片数量
    n_rows = math.ceil((img_h - SLICE_SIZE) / stride) + 1
    n_cols = math.ceil((img_w - SLICE_SIZE) / stride) + 1
    
    count = 0
    
    # 开始滑动窗口切片
    for row in range(n_rows):
        for col in range(n_cols):
            # 计算当前切片的坐标范围
            x1 = col * stride
            y1 = row * stride
            x2 = min(x1 + SLICE_SIZE, img_w)
            y2 = min(y1 + SLICE_SIZE, img_h)
            
            # 如果到了边缘，回退起始点以保证切片尺寸固定 (可选，这里保持尺寸固定有利于Batch处理)
            if x2 - x1 < SLICE_SIZE:
                x1 = max(0, x2 - SLICE_SIZE)
            if y2 - y1 < SLICE_SIZE:
                y1 = max(0, y2 - SLICE_SIZE)
                
            # 1. 裁剪图片
            slice_img = img[y1:y2, x1:x2]
            
            # 创建切片的 Shapely 几何框
            slice_poly = box(x1, y1, x2, y2)
            
            new_shapes = []
            
            # 2. 处理每一个标注
            for shape in shapes:
                label = shape['label']
                points = shape['points']
                
                # 只有 >= 3 个点才能构成多边形
                if len(points) < 3:
                    continue
                
                try:
                    obj_poly = Polygon(points)
                    
                    # 检查是否验证
                    if not obj_poly.is_valid:
                        obj_poly = obj_poly.buffer(0)
                        
                    # 计算交集 (Intersection)
                    if not slice_poly.intersects(obj_poly):
                        continue
                        
                    inter_poly = slice_poly.intersection(obj_poly)
                    
                    # 如果交集为空
                    if inter_poly.is_empty:
                        continue
                        
                    # 处理 MultiPolygon (如果一个楼盘被切成了两半，取最大的那一块或都取)
                    if inter_poly.geom_type == 'MultiPolygon':
                        polys_to_process = list(inter_poly.geoms)
                    elif inter_poly.geom_type == 'Polygon':
                        polys_to_process = [inter_poly]
                    else:
                        continue
                        
                    for p in polys_to_process:
                        # 过滤面积太小的残片
                        if p.area < MIN_AREA_THRESHOLD:
                            continue
                            
                        # --- 核心：坐标转换 (Global -> Local) ---
                        # 将全局坐标减去切片左上角坐标 (x1, y1)
                        local_poly = translate(p, xoff=-x1, yoff=-y1)
                        
                        # 提取坐标点
                        if local_poly.exterior:
                            local_points = list(local_poly.exterior.coords)
                            # 去掉最后一个重复点 (Shapely 多边形闭合时首尾相同)
                            local_points = local_points[:-1]
                            
                            new_shape = {
                                "label": label,
                                "points": local_points,
                                "group_id": shape.get("group_id", None),
                                "shape_type": "polygon",
                                "flags": shape.get("flags", {})
                            }
                            new_shapes.append(new_shape)
                            
                except Exception as e:
                    # 忽略极少数几何错误的标注
                    continue

            # 3. 只有当切片里有目标时，才保存 (过滤背景图)
            if len(new_shapes) > 0:
                save_name = f"{filename_base}_{row}_{col}"
                
                # 保存图片
                img_out_path = os.path.join(output_img_dir, save_name + ".png")
                cv2.imwrite(img_out_path, slice_img)
                
                # 保存 JSON
                json_out = {
                    "version": "4.5.6",
                    "flags": {},
                    "shapes": new_shapes,
                    "imagePath": save_name + ".png",
                    "imageData": None, # 不需要保存base64，节省空间
                    "imageHeight": slice_img.shape[0],
                    "imageWidth": slice_img.shape[1]
                }
                
                json_out_path = os.path.join(output_json_dir, save_name + ".json")
                with open(json_out_path, 'w') as f:
                    json.dump(json_out, f, indent=2)
                
                count += 1
                
    return count

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 查找所有的 json 文件
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    print(f"找到 {len(json_files)} 个标注文件，准备开始切片...")
    print(f"切片大小: {SLICE_SIZE}x{SLICE_SIZE}, 重叠率: {OVERLAP}")
    
    total_slices = 0
    
    for json_file in tqdm(json_files):
        # 寻找对应的图片 (支持 jpg 和 png)
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(INPUT_DIR, json_file)
        
        image_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            temp_path = os.path.join(INPUT_DIR, base_name + ext)
            if os.path.exists(temp_path):
                image_path = temp_path
                break
        
        if image_path:
            # 这里为了方便训练，把图片和json都输出到同一个文件夹
            # 也可以分开 output_dir/images 和 output_dir/annotations
            num = slice_single_image(image_path, json_path, OUTPUT_DIR, OUTPUT_DIR)
            total_slices += num
        else:
            print(f"警告: 找不到对应的图片 -> {json_file}")

    print(f"\n处理完成！")
    print(f"共生成了 {total_slices} 张切片训练图，保存在: {OUTPUT_DIR}")
    print("现在可以使用训练代码进行训练了。")

if __name__ == "__main__":
    main()