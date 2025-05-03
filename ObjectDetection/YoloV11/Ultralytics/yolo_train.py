from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model 
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    dataset_path = r"C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\ObjectDetection\YoloV11\YoloV11Dataset\data.yaml"

    # Train the model (configs documentation: https://docs.ultralytics.com/modes/train/#train-settings)
    model.tune(
        data=dataset_path,
        epochs=100, 
        imgsz=640, 
        batch=8, 
        val=True, 
        plots=True, 
        device="0",
        optimizer="AdamW"
    )
    
    results = model.train()

if __name__ == "__main__":
    main()