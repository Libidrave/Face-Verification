from ultralytics import YOLO 
import os
from glob import glob

detector = YOLO(glob("./*.pt")[0])

def crop(img_path : str, filename : str):
    """
    Crops an image file based on a given bounding box coordinates.
    Args:
        img_path (str): Path to the image file.
    Returns:
        str: Path to the cropped image file.
    """
    res_path = os.path.join("./face", filename)
    
    filename = filename.rsplit(".")[0].replace("/", "")
    result = detector.predict(img_path, conf=0.25, max_det=1)

    for results in result:
        results.save_crop("./", filename)

    return res_path
