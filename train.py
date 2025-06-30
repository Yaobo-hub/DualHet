import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
   # model = MyModel().cuda().half()  # 将模型转换为 Half 类型并移动到 GPU
    # 假设 model 是您的模型，input_data 是您的输入数据
    model = YOLO('yolov5s.yaml')
    #yolo11-C3k2-DSConv.yaml,yolo11-TwoBackbone-DynamicHead-yolo11-C3k2-DSConv.yaml
    model = model.cuda().half() # 续训yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
    # 假设 model 是您的模型，input_data 是您的输入数据
    #/root/autodl-tmp/autodl-tmp/YOLOv11/runs/train/exp2/weights/last.pt
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load('yolov11n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(data="ultralytics/cfg/datasets/MyData-cow.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                task='detect',
                cache=False,
                imgsz=640,
                epochs=600,
                single_cls=False,  # 是否是单类别检测
                batch=32,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='/root/runs/yolov5s-cow',
                name='exp',
                )

