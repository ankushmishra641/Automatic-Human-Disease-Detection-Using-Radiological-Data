# ____________________________________


import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model1 = load_model('brain_tumor_detector_my.h5', custom_objects={'KerasLayer': hub.KerasLayer})

model_test1 = load_model('Brain_test_Detector.h5', custom_objects={'KerasLayer': hub.KerasLayer})
# model_test = load_model('chest_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def model_predict_test1(img_path, model):
    test_image1 = image.load_img(img_path, target_size = (256,256))
    test_image1 = image.img_to_array(test_image1)
    test_image1 = test_image1 / 255
    test_image1 = np.expand_dims(test_image1, axis=0)
    result1 = (model.predict(test_image1)>0.5).astype("int32")
    result1

    return result1

def model_predict1(img_path, model):
    test_image = image.load_img(img_path, target_size = (150,150))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)


    result2 = model.predict(test_image)
    result2=result2.argmax()



    if result2==0:
        result2 = "glioma_tumor"
    if result2==1:
        result2 = "meningioma_tumor"
    if result2==2:
        result2 = "no_tumor"
    if result2==3:
        result2 = "pituitary_tumor"
    return result2

# Model saved with Keras model.save()
model2= load_model('Covid-19 Detector.h5', custom_objects={'KerasLayer': hub.KerasLayer})

model_test2 = load_model('chest_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})



def model_predict_test2(img_path, model):
    test_image1 = image.load_img(img_path, target_size = (256,256))
    test_image1 = image.img_to_array(test_image1)
    test_image1 = test_image1 / 255
    test_image1 = np.expand_dims(test_image1, axis=0)
    result1 = (model.predict(test_image1)>0.5).astype("int32")
    result1

    return result1
def model_predict2(img_path, model):
            test_image = image.load_img(img_path, target_size = (256,256))
            test_image = image.img_to_array(test_image)
            test_image = test_image / 255
            test_image = np.expand_dims(test_image, axis=0)

            acc=((model.predict(test_image)))
            acc=int(100-(acc*100))
            # if acc>=70:
            #     a_text="Having more Covid Symptoms,Consult to a Doctor"
            # if acc<=70 and acc>=50:
            #     a_text="Mild Covid Symptoms"
            # if acc<=50 and acc>=0:
            #     a_text="May get affected follow remedies"
           
            
            result2 = (model.predict(test_image)>0.5).astype("int32")
            result2


            if result2==True:
                result2 = "patient is Normal"
                
            else:
                result2 = f"patient has Covid-19 (Detected Disease Percentage : { acc })"

            return result2
        

# Model saved with Keras model.save()
model3 = load_model('kidney_stone_detection.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def model_predict3(img_path, model):
    test_image = image.load_img(img_path, target_size = (256,256))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)

    acc=((model.predict(test_image)))
    acc=int((acc*100))
    
    result = (model.predict(test_image)>0.5).astype("int32")
    result


    if result==1:
        result = f"Patient has kidney stone (Detected Disease Percentage : { acc })"
    else:
        result =  "Patient has NO kidney stone"

    return result


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/index1', methods=['GET'])
def index1():
    return render_template('index1.html')

@app.route('/index2', methods=['GET'])
def index2():
    return render_template('index2.html')

@app.route('/index3', methods=['GET'])
def index3():
    return render_template('index3.html')

@app.route('/predict1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads1', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        # preds = model_predict1(file_path, model1)
        # result = preds
        # return result
        
        test_result1 =model_predict_test1(file_path, model_test1)
        test_result2 =model_predict1(file_path, model1)
        if test_result1==True:
            result="Please give valid data"
        else:
            result=test_result2
        return result
    return None


@app.route('/predict2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads2', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction


        test_result1 =model_predict_test2(file_path, model_test2)
        test_result2 =model_predict2(file_path, model2)
        if test_result1==True:
            result="Please give valid data"
        else:
            result=test_result2
        return result
        
        
    return None

@app.route('/predict3', methods=['GET', 'POST'])
def upload3():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads3', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict3(file_path, model3)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False)
    