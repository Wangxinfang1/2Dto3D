import gradio as gr
import cv2
import os
import numpy as np
import trimesh
import torch
from shapely.geometry import Polygon
from shapely.validation import make_valid

# --- Detectron2 & PointRend Imports ---
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects import point_rend

# ================= 配置路径 (请修改这里) =================
CONFIG_PATH = "/home/wxf/output_unified_pointrend_aug/config.yaml"
MODEL_PATH = "/home/wxf/output_unified_pointrend_aug/model_final.pth"
# ========================================================

# -----------------------------------------------------------------------------
# 1. 初始化模型 (全局加载一次，避免每次请求都重新加载)
# -----------------------------------------------------------------------------
print("正在加载模型，请稍候...")
try:
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 强制设置推理分辨率 (保留细节)
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 4096
    
    predictor = DefaultPredictor(cfg)
    
    # 获取元数据用于可视化 (用于在图片上画框和标签)
    metadata = MetadataCatalog.get("site_train")
    metadata.thing_classes = ["building", "crane"] # 确保顺序正确
    
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请检查 config.yaml 和 model_final.pth 的路径是否正确。")
    predictor = None

# -----------------------------------------------------------------------------
# 2. 核心处理逻辑
# -----------------------------------------------------------------------------
def inference_and_generate(image, b_height, c_radius):
    """
    输入:
        image: numpy array (RGB)
        b_height: 楼盘高度 (float)
        c_radius: 塔吊半径 (float)
    输出:
        annotated_image: 检测结果图
        glb_path: 3D模型文件路径
    """
    if predictor is None:
        raise gr.Error("模型未加载，请检查后台日志。")
    
    if image is None:
        raise gr.Error("请先上传图片或拍摄照片。")

    # 1. 图像预处理 (Gradio 传入的是 RGB，OpenCV 需要 BGR)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 2. 推理
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    
    # 3. 生成 2D 可视化结果
    v = Visualizer(image, metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances)
    vis_img = out.get_image() # RGB format
    
    # 4. 提取数据用于 3D 建模
    pred_classes = instances.pred_classes.numpy()
    pred_masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    
    buildings_polygons = []
    cranes_centers = []
    
    for i, class_id in enumerate(pred_classes):
        # 过滤低置信度 (虽然模型已有阈值，这里可以再次把控)
        if scores[i] < 0.5: continue
        
        mask = pred_masks[i]
        
        if class_id == 0: # Building
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    epsilon = 0.003 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    if len(approx) >= 3:
                        buildings_polygons.append(approx.reshape(-1, 2))
                        
        elif class_id == 1: # Crane
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cranes_centers.append((cX, cY))

    # 5. 生成 3D 模型
    scene = trimesh.Scene()
    
    # A. 楼盘建模 (高度由滑块控制)
    for poly in buildings_polygons:
        # 坐标转换：图像Y轴向下，3D Y轴向上，所以取反
        poly_3d = poly.copy()
        poly_3d[:, 1] = -poly_3d[:, 1]
        
        try:
            shapely_poly = Polygon(poly_3d)
            if not shapely_poly.is_valid:
                shapely_poly = make_valid(shapely_poly)
                if shapely_poly.geom_type == 'MultiPolygon':
                     shapely_poly = max(shapely_poly.geoms, key=lambda a: a.area)
            
            if not shapely_poly.is_empty:
                # 使用用户输入的高度
                mesh = trimesh.creation.extrude_polygon(shapely_poly, height=b_height)
                mesh.visual.face_colors = [200, 200, 200, 255] # 混凝土灰
                scene.add_geometry(mesh)
        except:
            pass

    # B. 塔吊建模 (半径由滑块控制)
    for center in cranes_centers:
        cx, cy = center
        transform = np.eye(4)
        transform[0, 3] = cx
        transform[1, 3] = -cy 
        
        # 塔吊主体 (红色细柱子) - 高度设为楼盘高度+20
        tower_height = b_height + 50
        tower = trimesh.creation.cylinder(radius=5, height=tower_height)
        tower.visual.face_colors = [255, 50, 50, 255]
        
        # 修正圆柱体中心点 (默认圆柱中心在原点，需要向上平移一半高度)
        # trimesh cylinder 默认中心在 (0,0,0)，长度在Z轴
        # 我们需要先生成，再旋转到 Y 轴 (Gradio Viewer通常 Y-up) 或者 Z-up
        # 简单起见，这里不需要旋转，因为 extrude 也是沿 Z 轴的
        # 只需要调整 Z 位置让它立在地上
        tower.apply_translation([0, 0, tower_height/2 - b_height/2]) # 稍微调整位置
        tower.apply_transform(transform)
        scene.add_geometry(tower)
        
        # 覆盖范围 (半透明蓝色圆柱)
        range_h = 2
        range_cyl = trimesh.creation.cylinder(radius=c_radius, height=range_h)
        range_cyl.visual.face_colors = [0, 100, 255, 50] # RGBA
        range_cyl.apply_transform(transform)
        scene.add_geometry(range_cyl)

    # 6. 导出 GLB
    output_path = "temp_result.glb"
    scene.export(output_path)
    
    return vis_img, output_path

# -----------------------------------------------------------------------------
# 3. 构建 UI 界面
# -----------------------------------------------------------------------------
# 使用 Soft 主题，看起来更现代
theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="工程图纸 3D 重建系统") as demo:
    gr.Markdown(
        """
        # 🏗️ 工程图纸智能 3D 重建系统
        上传工程平面图或拍摄照片，系统将自动识别**楼盘**与**塔吊**，并生成可交互的 3D 场景。
        """
    )
    
    with gr.Row():
        # --- 左侧：输入区 ---
        with gr.Column(scale=1):
            # ================== 修改重点开始 ==================
            # 使用一个组件同时支持上传和拍照
            # sources=["upload", "webcam"] 允许用户选择上传文件或使用摄像头
            img_input = gr.Image(
                sources=["upload", "webcam"], 
                type="numpy", 
                label="上传平面图或拍照"
            )
            # ================== 修改重点结束 ==================
            
            gr.Markdown("### ⚙️ 3D 参数设置")
            slider_height = gr.Slider(minimum=10, maximum=1000, value=300, step=10, label="楼盘高度 (Building Height)")
            slider_radius = gr.Slider(minimum=50, maximum=1000, value=400, step=10, label="塔吊覆盖半径 (Crane Radius)")
            
            btn_run = gr.Button("🚀 开始检测并生成 3D 模型", variant="primary")

        # --- 右侧：输出区 ---
        with gr.Column(scale=2):
            with gr.Tab("2D 检测结果"):
                img_output = gr.Image(label="识别结果可视化")
            with gr.Tab("3D 交互模型"):
                model_output = gr.Model3D(
                    clear_color=[1.0, 1.0, 1.0, 1.0], 
                    label="3D 预览 (支持旋转/缩放)",
                    interactive=True
                )

    # --- 事件绑定 ---
    # 现在 img_input 无论来自上传还是拍照，都会正确传入
    btn_run.click(
        fn=inference_and_generate,
        inputs=[img_input, slider_height, slider_radius],
        outputs=[img_output, model_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)