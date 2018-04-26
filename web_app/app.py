from flask import Flask, render_template, request, jsonify
import pickle
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
from make_image import make_image
from make_model import make_model
from flask_bootstrap import Bootstrap

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return predict(filename)
    return render_template('upload.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict(filename = False):
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    if filename:
        filename = 'static/img/' + filename
        display_image, tot_time = make_image(filename, graph)
        tot_time = "Total load time:" +str(tot_time)
    else:
        display_image = 'static/detected_img/three_dogs_detected.png'
        tot_time = ''
    print(display_image)
    return render_template('predict.html', user_image=display_image,
                            tot_time=tot_time
                            )


if __name__ == '__main__':
    bootstrap = Bootstrap(app)
    graph = make_model()
    app.run(host='0.0.0.0', debug=True)
