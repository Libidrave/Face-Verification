import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import os
import io
import time
from transformers import pipeline
from pathlib import Path
from PIL import Image

from module.crop import crop
from module.findCosine import find_cosine_distance
from module.inference import recognize
from module.preprocessing import resize_image

app = FastAPI()

CONFIG = {"UPLOAD FOLDER": "static/"}
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model directly
pipebob = pipeline("image-classification", model="./model/blurorbokeh/", image_processor="./model/blurorbokeh/")
pipecon = pipeline("image-classification", model="./model/cartoonornot/", image_processor="./model/cartoonornot/")

# Remove all file on folder
folder_paths = [Path("./static"), Path("./face")]
for folder_path in folder_paths:
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()

@app.get("/")
async def home():
    return {
        "status_code": 200,
        "message": "Server is up",
        "data" : None
        }, 200

@app.post("/verify")
async def verify(img1 : UploadFile, img2 : UploadFile):
    img1_path = None
    img2_path = None

    # Checking if the file exists
    if img1.filename == "":
        return "No selected file", 400
    if img2.filename == "":
        return "No selected file", 400
    
    # Checking image extension  
    if img1 and allowed_file(img1.filename):
        contents1 = await img1.read()
        new_img1 = Image.open(io.BytesIO(contents1))
        img1_path = os.path.join(CONFIG["UPLOAD FOLDER"], img1.filename)
        new_img1.save(img1_path)
    else:        
        return JSONResponse({
            "status_code": 400,
            "message": "Invalid file type, only png, jpg, jpeg are allowed",
            "data": None
        },400)
    
    if img2 and allowed_file(img2.filename):
        contents2 = await img2.read()
        new_img2 = Image.open(io.BytesIO(contents2))
        img2_path = os.path.join(CONFIG["UPLOAD FOLDER"], img2.filename)
        new_img2.save(img2_path)
    else:        
        return JSONResponse({
            "status_code": 400,
            "message": "Invalid file type, only png,jpg,jpeg are allowed",
            "data": None
        },400)

    # Blur Detection
    try:
        prediction = pipebob(img1_path, function_to_apply='softmax')
        y_pred = prediction[0]["label"]

        if y_pred == "Blur":
            return JSONResponse({
                "status_code": 400,
                "message": "Image Detected as Blurred, Try Again!!",
                "data": None
            },400)
    except Exception as e:
        return JSONResponse({
            "status_code": 400,
            "message": f"Error occurred during blur detection because of {str(e)}",
            "data" : None
        },400)
        
    # Cartoon Detection
    try:
        prediction = pipecon(img2_path, function_to_apply='sigmoid')
        y_pred = prediction[0]["label"]

        if y_pred == "cartoon":
            return JSONResponse({
                "status_code": 400,
                "message": "Image Detected as Cartoon, Try Again!!",
                "data" : None
            },400)
    except Exception as e:
        return JSONResponse({
            "status_code": 400,
            "message": f"Error occurred during cartoon detection because of {str(e)}",
            "data" : None
        },400)

    # Face Verification
    try:
        start_time = time.time()

        img1_path = crop(img1_path, img1.filename)
        img2_path = crop(img2_path, img2.filename)
        img1 = resize_image(img1_path)
        img2 = resize_image(img2_path)

        output1 = recognize(img1)
        output2 = recognize(img2)
        distance = find_cosine_distance(output1, output2)
        result = None
        end_time = time.time()
        final_time = end_time - start_time
        final_time *= 1000

        if distance <= 0.40:
            result = {
                "model_name": "Facenet",
                "threshold": 0.40,
                "distance": distance,
                "verified": True,
                "time": f"{final_time:.2f} ms"
            }
        else:
            result = {
                "model_name": "Facenet",
                "threshold": 0.40,
                "distance": distance,
                "verified": False,
                "time": f"{final_time:.2f} ms"
            }

        result['verified'] = bool(result['verified'])

        if result['verified'] == True:
            return JSONResponse({                
                    "status_code": 200,
                    "message": "Face Verified Successfully",
                    "data" : result
            }, 200)
                
        else:
            return JSONResponse({
                    "status_code": 400,
                    "message": "Face Verification Failed, Please Upload Your Photo Again!",
                    "data" : None
            }, 400)
                        
                
    except Exception as e:
        return JSONResponse({
            "status_code": 400,
            "message": f"Error occurred during verification because of {str(e)}",
            "data" : None
        },400)


