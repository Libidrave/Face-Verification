import onnxruntime as rt
import os
from glob import glob
sess = rt.InferenceSession(glob("./*.onnx")[0])

def recognize(img):
    input = sess.get_inputs()[0].name
    output = sess.get_outputs()[0].name

    result = sess.run([output], {input: img})[0][0].tolist()
    return result