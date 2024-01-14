import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle
import os
import imutils

MODEL_NAME_INPUT = "animal.model"
LABEL_NAME_INPUT = 'lb.pickle'


def upload_and_predict():
    # 使用filedialog让用户选择一个图片文件
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    model, lb = load_label_and_model(MODEL_NAME_INPUT, LABEL_NAME_INPUT)
    if model == None or lb == None:
        return
    
# 使用OpenCV加载图片
    image_cv = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)  # 从BGR转换到RGB
    output = Image.fromarray(image_rgb)  # 将OpenCV图像转换为PIL图像
    fixed_height = 300
    original_width, original_height = output.size
    aspect_ratio = original_width / original_height
    new_width = int(fixed_height * aspect_ratio)

    output = output.resize((new_width, fixed_height))  # 重新调整图片大小以适应GUI

    tk_image = ImageTk.PhotoImage(output)
    label_image.config(image=tk_image)
    label_image.image = tk_image

    # pre-process the image for classification
    input = cv2.resize(image_rgb, (96, 96))
    input = input.astype("float") / 255.0
    input = img_to_array(input)
    input = np.expand_dims(input, axis=0)


    # 使用模型预测图片类别
    predicted_class = predict_single_image(model, lb, input)
    label_result.config(text=f"Predicted class: {predicted_class}")

def load_label_and_model(model_name_input, label_namme_input):
    if(os.path.isfile(label_namme_input) == False):
        print("[INFO] label not found")
        return None, None
    if(os.path.isdir(model_name_input) == False):
        print("[INFO] model folder not found")
        return None, None
    # load the trained convolutional neural network and the label
    # binarizer
    print("[INFO] loading network...")
    model = load_model(model_name_input)
    lb = pickle.loads(open(label_namme_input, "rb").read())
    return model, lb


def predict_single_image(model, lb, image):
        # classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    
    return label

    

if __name__ == '__main__':


    # GUI (Graphic User Interface)
    window = tk.Tk()
    window.geometry("800x600")
    window.title("Animal Decetor")

    # 上传按钮
    btn_upload = tk.Button(window, text="Upload an image", command=upload_and_predict)
    btn_upload.pack(pady=20)


    # 图片显示标签
    label_image = tk.Label(window)
    label_image.pack(pady=20)

    # 预测结果显示标签
    label_result = tk.Label(window, text="Predicted class will appear here", font=("Arial", 12))
    label_result.pack(pady=20)

    window.mainloop()




