import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
   # model = MyModel().cuda().half() 
    model = YOLO('yolov5s.yaml')
    #yolo11-C3k2-DSConv.yaml,yolo11-TwoBackbone-DynamicHead-yolo11-C3k2-DSConv.yaml
    model = model.cuda().half() # 续训yaml文件的地方改为lats.pt的地址
    #/root/autodl-tmp/autodl-tmp/YOLOv11/runs/train/exp2/weights/last.pt
    # model.load('yolov11n.pt') 
    model.train(data="ultralytics/cfg/datasets/MyData-cow.yaml",
                task='detect',
                cache=False,
                imgsz=640,
                epochs=600,
                single_cls=False,  # 是否是单类别检测
                batch=32,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD 
                # resume=, # 续训的话这里填写True
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='/root/runs/yolov5s-cow',
                name='exp',
                )

