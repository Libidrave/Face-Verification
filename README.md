# Face Verification Using Python & FastAPI
Hello Everyone, this repository is used to store any file that related to this project

# What You Need
1. Programming Language : Python (3.8 - 3.10)
2. Library : you can install all the library using pip in your terminal `pip install -r requirements.txt`

# Project Description
This project is really inspired from this repository [DeepFace](https://github.com/serengil/deepface) and i added [CartoonOrNot Classifier](https://github.com/Libidrave/CartoonOrNot) and [BlurOrBokeh Classifier](https://github.com/Libidrave/BlurOrBokeh) \
to make sure the input image is not ai generated and not a blurred image. So how does this project work?
1. You need to upload 2 images, the first one is your ID card (KTP) and the second one is your selfie photo.
2. The first **CartoonOrNot** model trying to classifying your selfie photo, if its not an ai generated image, then
3. The second **BlurOrBokeh** model trying to classifying yout selfie photo, if its not a blurred image, then
4. All the images you upload, will be cropped to match the face detected by the face detector model.
5. The last step is, images that have been cropped will be paired and their similarity is calculated using cosine similarity.
