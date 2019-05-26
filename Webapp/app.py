from flask import Flask
from flask import request
from scipy.misc import imshow, imsave, imread, imresize
import cv2
from werkzeug import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
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

import subprocess
from subprocess import Popen, PIPE
from flask import Flask, render_template, send_from_directory
app = Flask(__name__)

@app.route('/input/')
def take_input():
   return render_template('hack.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.filename = 'test-volume-0.nii'
      f.save('./saved_data/'+f.filename)

      session = subprocess.Popen(['./copy_to_biometric.sh'], stdout=PIPE, stderr=PIPE)
      stdout, stderr = session.communicate()

      session = subprocess.Popen(['./helper.sh'], stdout=PIPE, stderr=PIPE)
      stdout, stderr = session.communicate()
   return send_from_directory('./', 'output.nii', as_attachment=True)
   
      # if stderr:
      #    raise Exception("Error"+str(stderr))
   # return "Uploaded"

# @app.route('/servertest')
# def servertest():
#    session = subprocess.Popen(['./helper.sh'], stdout=PIPE, stderr=PIPE)
#    stdout, stderr = session.communicate()
#    # if stderr:
#       # raise Exception("Error"+str(stderr))
#    return stdout.decode('utf-8')

# @app.route('/download')
# def download_data():
#    return send_from_directory('./', 'output.nii', as_attachment=True)
    
if __name__ == '__main__':
   app.run(debug = True)