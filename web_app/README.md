# Web App guide

It's a pretty ~simple web app~ really complicated and awesome web app <sup><sub>please hire me</sup></sub>. This allows users to upload their own photos and get them detected. It has 3 tabs (wow!), a splash/welcome page, an upload page, and one that shows them their image. It also has an example image, whose deteced image is preloaded. 

To run this app, you'll need to clone this directory, make a directory called tensorflow_things (or clone my entire repo). And clone the [tensorflow/models](https://github.com/tensorflow/models) repo into the tensorflow_things directory. Navigate to tensorflow_things/models/research and paste this (or put it in your bash profile): ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim```. Then just go back to the web_app directory and type ```python app.py``` in your terminal. You'll also want to change the weights file to the correct one. 

I used a template for the basic layout, then changed some things. I probably could have made the code a bit cleaner, but had a day to make this, so I didn't. 

### An overview of how it works

I'm not explaining flask to you here, or HTML, or CSS, just what my code does. First, the app builds the model, from make_model.py. Then it loads the webpage. When a user selects and submit a file, it saves it to the static/img folder. It then calls make_image.py on that image, which saves the deteced image to the static/detected_img folder, and displays it on the results page. Here are the three pages:

![homepage](https://github.com/MasonCaiby/Boulder_AI_CPW_study/blob/master/Screen%20Shot%202018-04-26%20at%202.58.46%20PM.png)
![upload page](https://github.com/MasonCaiby/Boulder_AI_CPW_study/blob/master/Screen%20Shot%202018-04-26%20at%202.58.59%20PM.png)
![results page](https://github.com/MasonCaiby/Boulder_AI_CPW_study/blob/master/Screen%20Shot%202018-04-26%20at%202.59.07%20PM.png)
