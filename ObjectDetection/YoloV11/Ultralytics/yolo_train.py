from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    dataset_path = r"C:\Users\User\Downloads\RealTimeAIMalaysianCarplateDetection\final-dataset.v6i.yolov11\data.yaml"

    # Train the model
    results = model.train(data=dataset_path, epochs=100, imgsz=640, val=False)

if __name__ == "__main__":
    main()