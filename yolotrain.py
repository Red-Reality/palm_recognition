from ultralytics import YOLO
if __name__ == '__main__':
# 从头开始创建一个新的YOLO模型
    model = YOLO('yolov8s.yaml')

    # 加载预训练的YOLO模型（推荐用于训练）
    model = YOLO('yolov8s.pt')

    # 使用“coco128.yaml”数据集训练模型3个周期
    results = model.train(data='./voc.yaml', epochs=30,imgsz=512)

    # # 评估模型在验证集上的性能
    # results = model.val()
    #
    # # 使用模型对图片进行目标检测
    # results = model('https://ultralytics.com/images/bus.jpg')

    # 将模型导出为ONNX格式
    model = YOLO('runs/detect/train9/weights/best.pt')  # load a custom trained model

# Export the model
    model.export(format='onnx')