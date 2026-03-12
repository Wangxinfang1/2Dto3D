import cv2
import os
import numpy as np
import trimesh
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from shapely.geometry import Polygon
# --- 【关键修改】引入 PointRend 模块 ---
from detectron2.projects import point_rend 

# ---------------------------------------------------
# 1. 加载训练好的 PointRend 模型
# ---------------------------------------------------
def get_predictor():
    cfg = get_cfg()
    
    # --- 【关键修改】必须先注册 PointRend 配置 ---
    point_rend.add_pointrend_config(cfg)
    
    # 读取训练保存的配置文件 (请确认路径是否对应刚才训练的输出目录)
    # 假设你上一轮训练输出在 ./output_unified_pointrend
    config_path = "/home/wxf/output_unified_pointrend_aug/config.yaml"
    model_path = "/home/wxf/output_unified_pointrend_aug/model_final.pth"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    
    # 置信度阈值 (PointRend通常比较自信，可以设高一点过滤误检)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    
    # 强制不调整大小，或者设置得很大，以保留工程图细节
    # 如果显存不够 (OOM)，请适当调小 MAX_SIZE_TEST
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 4096 
    
    return DefaultPredictor(cfg)

# ---------------------------------------------------
# 2. 推理并提取 3D 建模所需数据
# ---------------------------------------------------
def process_image(image_path):
    print(f"开始加载模型并处理图片: {image_path}")
    predictor = get_predictor()
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    outputs = predictor(img)
    
    # 获取推理结果
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.numpy()
    pred_masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    
    buildings_polygons = []
    cranes_centers = []
    
    print(f"检测到 {len(pred_classes)} 个目标")
    
    # 遍历所有检测到的对象
    for i, class_id in enumerate(pred_classes):
        score = scores[i]
        mask = pred_masks[i]
        
        # --- 如果是楼盘 (class_id == 0) ---
        if class_id == 0: 
            # 将掩码转换为轮廓坐标
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # PointRend 生成的边缘通常很干净，但也可能有一些微小的噪点
                # 过滤掉极小的区域
                if area > 50: 
                    # 简化轮廓：PointRend 出来的点很密，适当简化有助于建模性能
                    # epsilon 越小，保留的细节越多；越大，线条越直
                    epsilon = 0.003 * cv2.arcLength(cnt, True) 
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # 确保是多边形（至少3个点）
                    if len(approx) >= 3:
                        buildings_polygons.append(approx.reshape(-1, 2))
                        
        # --- 如果是塔吊 (class_id == 1) ---
        elif class_id == 1:
            # 计算掩码的重心
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cranes_centers.append((cX, cY))
                print(f"  -> 发现塔吊 (置信度: {score:.2f}) at ({cX}, {cY})")
    
    return buildings_polygons, cranes_centers

# ---------------------------------------------------
# 3. 生成 3D 场景
# ---------------------------------------------------
def generate_3d(buildings, cranes):
    scene = trimesh.Scene()
    
    print(f"准备生成 3D 模型：楼盘数量={len(buildings)}, 塔吊数量={len(cranes)}")

    # 生成楼盘
    for idx, poly in enumerate(buildings):
        # 1. 翻转Y轴
        poly_3d = poly.copy()
        poly_3d[:, 1] = -poly_3d[:, 1]
        
        try:
            shapely_poly = Polygon(poly_3d)
            
            # --- 【核心修复开始】 ---
            if not shapely_poly.is_valid:
                from shapely.validation import make_valid
                shapely_poly = make_valid(shapely_poly)
                
                # 情况 A: 修复后变成了 "大杂烩" (GeometryCollection)
                # 这种情况下，我们要遍历里面的东西，只保留多边形
                if shapely_poly.geom_type == 'GeometryCollection':
                    polys = []
                    for geom in shapely_poly.geoms:
                        if geom.geom_type in ['Polygon', 'MultiPolygon']:
                            polys.append(geom)
                    
                    if len(polys) > 0:
                        # 如果有多个多边形，取面积最大的那个
                        shapely_poly = max(polys, key=lambda a: a.area)
                    else:
                        # 如果大杂烩里全是线和点，没有面，那就没法建模，跳过
                        print(f"  -> 第 {idx+1} 栋楼盘修复后无有效多边形，跳过")
                        continue

                # 情况 B: 修复后变成了多个多边形 (MultiPolygon)
                if shapely_poly.geom_type == 'MultiPolygon':
                     shapely_poly = max(shapely_poly.geoms, key=lambda a: a.area)
            # --- 【核心修复结束】 ---

            # 2. 拉伸建模
            if not shapely_poly.is_empty and shapely_poly.area > 1:
                mesh = trimesh.creation.extrude_polygon(shapely_poly, height=300)
                mesh.visual.face_colors = [200, 200, 200, 255] # 灰色
                scene.add_geometry(mesh)
                # print(f"  -> 第 {idx+1} 栋楼盘生成成功")
            
        except Exception as e:
            # 打印更详细的错误，方便排查
            print(f"  -> 第 {idx+1} 栋楼盘生成失败，原因: {type(e).__name__}: {e}") 
            
    # 生成塔吊 (保持不变)
    for center in cranes:
        cx, cy = center
        transform = np.eye(4)
        transform[0, 3] = cx
        transform[1, 3] = -cy 
        
        tower = trimesh.creation.cylinder(radius=8, height=80)
        tower.visual.face_colors = [255, 50, 50, 255]
        tower.apply_transform(transform)
        scene.add_geometry(tower)
        
        range_cyl = trimesh.creation.cylinder(radius=400, height=5)
        range_cyl.visual.face_colors = [0, 100, 255, 50] 
        range_cyl.apply_transform(transform)
        scene.add_geometry(range_cyl)
    
    # 保存
    output_filename = "result_pointrend1.glb"
    scene.export(output_filename)
    print(f"成功！3D模型已保存为 {output_filename}，请下载查看。")

# ---------------------------------------------------
# 主程序
# ---------------------------------------------------
if __name__ == "__main__":
    # 请替换为实际的测试图片路径
    img_path = "/home/wxf/dataset/test/test1.png" 
    
    if os.path.exists(img_path):
        b_polys, c_centers = process_image(img_path)
        generate_3d(b_polys, c_centers)
    else:
        print(f"错误：测试图片不存在 -> {img_path}")