import os
from flask import Flask, make_response,request,session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
from PIL import Image, ImageOps
from numpy import array
from handleImage import  classify
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def index():
    #t=classify('img1.jpg')
    return "t"
@app.route('/api/newpic',methods = ['POST'])
def newpic():
    target=os.path.join('/','test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['newpic'] 
    #img=open(file.read(),'rb')
    #frame = cv2.imdecode(np.frombuffer(file.read(), numpy.uint8), cv2.IMREAD_COLOR)
    SIZE = (299, 299)
    image = Image.open(io.BytesIO(file.read()))
    image = ImageOps.fit(image, SIZE)
    # #print(image)
    image.show()
    # image.thumbnail(MAX_SIZE)
    image=array(image)
# # creating thumbnail
#     image.save('pythonthumb.jpg')
#     image.show()
    print(image.shape)
    res_cap=classify(image)
    print(res_cap)
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    res = make_response(res_cap)
    return res
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)