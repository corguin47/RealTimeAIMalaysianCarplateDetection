from ultralytics import YOLO

def main():
    # Load pretrained model to fine-tune (downloadable from https://github.com/ultralytics/ultralytics?tab=readme-ov-file)
    model = YOLO("yolo11n.pt")

    dataset_path = r"D:\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\YoloV11Dataset\data.yaml"

    # Train the model (configs documentation: https://docs.ultralytics.com/modes/train/#train-settings)
    model.tune(
        data=dataset_path,
        epochs=100, 
        imgsz=640, 
        batch=8, 
        val=True, 
        plots=True, 
        device="0",
        optimizer="AdamW",
    )
    
    results = model.train(plots=True)

if __name__ == "__main__":
    main()