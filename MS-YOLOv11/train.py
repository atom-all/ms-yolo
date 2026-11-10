import warnings, os

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml') # YOLO11
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='/root/dataset/dataset_visdrone/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                # device='0,1', 
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, 
                # amp=False, 
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
