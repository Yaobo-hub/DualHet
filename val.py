import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/runs/yolov5s-cow/exp/weights/best.pt')
    #/root/runs/TwoBackbone-DynamicHead-HetConv-2-train/exp3/weights/best.pt
    #autodl-tmp/autodl-tmp/YOLOv11/runs/train/exp2/weights/last.pt
    #runs/yolov5-train/exp/weights/last.pt
    model.val(data=r'/root/autodl-tmp/YOLOv11/ultralytics/cfg/datasets/MyData-cow.yaml',
              split='val',
              imgsz=640,
              batch=32,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/cow-val',
              name='exp',
              conf=0.75,
              )