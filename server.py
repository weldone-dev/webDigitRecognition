from threading import Thread

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

from PIL import Image
from io import BytesIO

import base64, os
import numpy  as np

model = load_model('my_model.h5')
app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('num_img')
def handle_message(message):
    img_data = base64.b64decode(str(message['data'][22:]))
    with open("imageToSave.png", "wb") as fh:
        fh.write(bytes(img_data))
    
    if os.path.exists("imageToSave.png"):
        im = Image.open("imageToSave.png")
        rgb_im = im.convert('RGB')
        rgb_im.save('imageToSave.jpg')
        os.remove("imageToSave.png")

    img = image.load_img("imageToSave.jpg", False, color_mode="grayscale", target_size=(28,28))
    x = image.img_to_array(img)
    x /= 255
    x = np.around(x)
    input_arr = np.array([x])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    emit('out_num', {'data': list(np.array(predictions[0])).index(max(predictions[0]))})
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
if __name__ == '__main__':
    socketio.run(app)
