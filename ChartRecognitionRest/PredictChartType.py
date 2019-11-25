'''
Created on Dec. 4, 2018

@author: mohammedmostafa
'''
import numpy as np
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request
from flask import render_template
from PIL import Image
import io

# creates a Flask application, named app
app = Flask(__name__)

dict_keys = ['bar_chart', 'bubble_chart', 'pie_chart', 'radar_chart', 'treemap_chart']


#default sizes
Image_Width = 100
Image_Height = 100

model = load_model("./model/CNDetector_transfer.h5")
model.summary()
# NOW WE LOAD THE PRE_TRAINED MODEL
model0 = load_model("./model/model.h5")
model0.summary()
model1 = load_model("./model/model1.h5")
model1.summary()
model2 = load_model("./model/model2.h5")
model2.summary()
model3 = load_model("./model/model3.h5")
model3.summary()
model4 = load_model("./model/model4.h5")
model4.summary()

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = xception.preprocess_input(image)
    # return the processed image
    return image

@app.route('/predict', methods=['POST'])
def upload():
    #files={'file': ('test.jpg', img, content_type)}
    if request.files.get("photo"):
        # read the image in PIL format
        src_image = request.files["photo"].read()
        src_image = Image.open(io.BytesIO(src_image))
        # preprocess the image and prepare it for classification
        x = prepare_image(src_image, target=(Image_Width, Image_Height))
        features_x = model0.predict(x)
        features_x1 = model1.predict(x)
        features_x2 = model2.predict(x)
        features_x3 = model3.predict(x)
        features_x4 = model4.predict(x)
        all_features = np.concatenate((features_x, features_x1,features_x2,features_x3,features_x4), axis=1)
        predictions = model.predict(all_features)
        row_index = predictions.argmax(axis=1)[0]
        return dict_keys[row_index]
    return "error"

@app.route("/")
def entryPage():
    return render_template('index.html')

# run the application
if __name__ == "__main__":
    app.run(debug=True)



