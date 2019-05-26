from flask import Flask
from flask import request
from scipy.misc import imshow, imsave, imread, imresize
import cv2
from werkzeug.datastructures import ImmutableMultiDict
import os
app = Flask(__name__)


# @app.route('/input', methods = ['POST'])
# def takeInput():
#     files = request.files
#     file1 = files['file']
#     file1.save('input.jpg')
#     convex.createSegments()
#     z = apiform.takeImageAndGiveText()
#     return z

# if __name__ == '__main__':
#     app.run()

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/hello/')
def hello_name(user):
   return render_template('../hack.html', name = user)

if __name__ == '__main__':
   app.run(debug = True)