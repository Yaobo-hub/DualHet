import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('/root/runs/yolov5s-cow/exp/weights/best.pt') # select your model.pt path
    #model = YOLO('/root/runs/TwoBackbone-DynamicHead-HetConv-2-train/exp3/weights/last.pt') # select your model.pt path
    #model = YOLO('/root/runs/yolov5s-train/exp2/weights/last.pt') # select your model.pt path
    #model = YOLO('/root/runs/TwoBackbone-RepGELAN(high)-ASFFHead-train/exp4/weights/last.pt') # select your model.pt path
    model = YOLO('/root/runs/C2PSA-iEMA-train/exp/weights/last.pt') # select your model.pt path
    
    #autodl-tmp/autodl-tmp/YOLOv11/runs/train/exp2/weights/last.pt
    #/root/runs/TwoBackbone-DynamicHead-HetConv-2-train/exp3/weights/best.pt
    #runs/yolov5s-train/exp2/weights/last.pt
    #runs/yolov5-train/exp/weights/last.pt
    #runs/TwoBackbone-RepGELAN(high)-ASFFHead-train/exp4/weights/last.pt
    #runs/C2PSA-iEMA-train/exp/weights/last.pt
    model.predict(source='/root/autodl-tmp/YOLOv11/datasets/chickendata/images/train/1224.jpg',
                  imgsz=640,
                  project='/root/runs/detect-1224',
                  name='exp',
                  save=True,
                  conf=0.6,
                  # classes=0, 是否指定检测某个类别.
                )