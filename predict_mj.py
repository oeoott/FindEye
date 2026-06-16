from ultralytics import YOLO
import os

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def predict_image(model, image_path):
    results = model(image_path)
    
    predictions = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            predictions.append({
                "class": class_name,
                "confidence": confidence,
                "image_path": image_path
            })
    
    return predictions

def predict_folder(model, folder_path):
    image_extensions = [".jpg", ".jpeg", ".png"]
    all_predictions = []
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(folder_path, filename)
            preds = predict_image(model, image_path)
            all_predictions.extend(preds)
    
    return all_predictions
