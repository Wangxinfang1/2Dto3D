import os
import cv2
import json
import copy
import torch
import numpy as np
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

# --- 1. 引入 PointRend 模块 ---
from detectron2.projects import point_rend
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
setup_logger()

# ---------------------------------------------------
# 1. 定义数据增强 Mapper (这是数据增强的核心)
# ---------------------------------------------------
def custom_mapper(dataset_dict):
    """
    自定义的数据读取和增强函数。
    每次训练迭代读取图片时，都会经过这里进行随机变换。
    """
    # 深拷贝，避免修改原始数据
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # 读取图片 (格式为 BGR)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # --- 定义增强策略 ---
    transform_list = [
        # 1. 随机旋转 (-180 到 180 度)：工程图纸没有方向性，旋转增强最重要！
        T.RandomRotation(angle=[-180, 180], expand=False),
        
        # 2. 随机水平翻转
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        
        # 3. 随机垂直翻转
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        
        # 4. 随机亮度与对比度 (模拟不同的扫描质量)
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        
        # 5. (可选) 随机裁剪：如果你的显存吃紧，或者想让模型更关注局部细节，可以开启
        # 如果你已经手动切片了，这里可以关掉或者只做轻微裁剪
        # T.RandomCrop("absolute", (800, 800)) 
    ]
    
    # 执行变换
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    # 将变换同时也应用到标注 (Mask 和 BBox) 上
    # 例如：图片旋转了90度，那么Mask的坐标也必须跟着转90度
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    
    # 转换成 Detectron2 需要的 Instance 格式
    instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")
    
    # 过滤掉因为裁剪或旋转后变为空的框
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
    
    return dataset_dict

# ---------------------------------------------------
# 2. 自定义 Trainer (为了挂载上面的 custom_mapper)
# ---------------------------------------------------
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # 使用自定义的 mapper
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# ---------------------------------------------------
# 3. 数据加载函数 (保持不变)
# ---------------------------------------------------
def get_site_dicts(img_dir):
    json_files = [f for f in os.listdir(img_dir) if f.endswith(".json")]
    dataset_dicts = []
    
    for idx, json_file in enumerate(json_files):
        try:
            with open(os.path.join(img_dir, json_file)) as f:
                imgs_anns = json.load(f)

            record = {}
            filename = os.path.join(img_dir, imgs_anns["imagePath"])
            
            if not os.path.exists(filename):
                print(f"Warning: {filename} not found, skipping.")
                continue

            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            for shape in imgs_anns["shapes"]:
                label = shape["label"]
                points = shape["points"]
                px = [x[0] for x in points]
                py = [x[1] for x in points]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                if label == "building":
                    category_id = 0
                elif label == "crane":
                    category_id = 1
                else:
                    continue 

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": category_id,
                }
                objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
            
    return dataset_dicts

# ---------------------------------------------------
# 4. 注册数据集
# ---------------------------------------------------
data_path = "/home/wxf/dataset/train_raw"  
if "site_train" in DatasetCatalog.list():
    DatasetCatalog.remove("site_train")
DatasetCatalog.register("site_train", lambda: get_site_dicts(data_path))
MetadataCatalog.get("site_train").set(thing_classes=["building", "crane"])

# ---------------------------------------------------
# 5. 配置与训练
# ---------------------------------------------------
if __name__ == "__main__":
    cfg = get_cfg()
    
    # --- PointRend 配置 ---
    point_rend.add_pointrend_config(cfg)
    
#    cfg.merge_from_file(model_zoo.get_config_file("InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"))
#    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    config_path = "/home/wxf/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        cfg.merge_from_file(config_path)
    else:
        # 如果找不到，手动去 GitHub 下载这个 yaml 文件放到项目目录下也行
        raise FileNotFoundError(f"找不到配置文件: {config_path}，请检查 detectron2 源码路径")

    # --- 【修改 2】使用直接链接加载权重 ---
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    
    cfg.DATASETS.TRAIN = ("site_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    
    # --- RTX 5090 优化参数 ---
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.IMS_PER_BATCH = 2   # 批次大小
    cfg.SOLVER.BASE_LR = 0.0005     # 学习率
    cfg.SOLVER.MAX_ITER = 5000     # 迭代次数
    cfg.SOLVER.STEPS = (2000, 2500)
    
    # PointRend 参数
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    num_classes = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_classes 

    # --- 针对工程图纸的 Anchor 和 尺寸 ---
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,) 
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MASK_FORMAT = "bitmask"  # 告诉模型输入数据是位图格式
    cfg.OUTPUT_DIR = "./output_unified_pointrend_aug"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- 使用 CustomTrainer 启动训练 ---
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 保存配置
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
        
    print(f"训练完成！输出目录: {cfg.OUTPUT_DIR}")