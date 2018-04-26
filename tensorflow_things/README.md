#Directory Guide: tensorflow_things

This directory is for training and running the model, I moved some scripts from tensorflow's models/... repo. I did this so I could call them a little easier, but mainly because I changed them a little, and wanted to be able to save those changes to git. To use this:

1. Clone the [tensorflow models/research repo](https://github.com/tensorflow/models) into the cloned Boulder_AI_CPW_study/tensorflowthings (this) directory. 
  follow the install instructions [from tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

2. Put your training images in a folder called ```.../tensorflow_things/image_annotations/images```

3. Put your training annotations in a folder called ```.../tensorflow_things/image_annotations/annotations```. The name of the object for each bounding box can be in the xml file OR the folder name.

4. Navigate to models/research in your terminal and copy/paste this code into term (you can also add it to your Bash Profile): 

```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim```

5. Run the ```xml_to_csv.py``` script. This will create a ```test.csv``` and ```train.csv``` file - the files wil be mixed but you must make sure you have the correct number of images for each class (e.g. balanced classes). How you obtain this is up to you: you can up sample, down sample, or just get the right number of images for each object.

6. Copy and paste both of these commands into your terminal (from ```tensorflow_things/```). 
```python generate_tfrecord.py --csv_input=image_annotations/train.csv --output_path=train2.record```
``` python generate_tfrecord.py --csv_input=image_annotations/test.csv --output_path=test2.record```

7. Sweet now you have your tfrecord files. Now we have to copy some things into the tensorflow_things directory. You'll adopt this for whichever model you wish to use (I used ```faster_rcnn_inception_resnet_v2_atrous_coco```).

    a. the config file for the model you want to use from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). 
  
    b. the tar'ed directory of the last checkpoint from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
8. Change the config file to have the correct number of classes (for the faster rcnn it is num_classes) here:
```
model {
  faster_rcnn {
    num_classes: 3
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
```
        
9. Change the filepaths to point to the correct location, they are in all caps. For the [faster_rcnn](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_coco.config) I used they are:
    a. line 108: ```fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"``` to ```fine_tune_checkpoint: "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/model.ckpt"```
        
    b. line 123: ```input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"``` to```input_path: "train.record"```
       
    c. line 125: ```label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"``` to```label_map_path: "object-detection.pbtxt"```
       
    d. line 137: ```input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"``` to```input_path: "test.record"```
       
    e. line 139: ```label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"``` to```label_map_path: "object-detection.pbtxt"```

10. You should now be good to run your train your model with this line:
    ```
    python models/research/object_detection/train.py --logtostderr --train_dir=training_faster_smaller_lr_1_3/ --pipeline_config_path=faster_rcnn_inception_resnet_v2_atrous_coco.config
    ```
    change the train_dir to your desired directory, and the pipeline_config_path to the path to your altered config file
    
11. Sweet. Model trained. Now export the model to a proto:
```
  python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph.pb
```
    change the pipeline_config_path, trained_checkpoint_prefic and output_directory as needed.
    
12. You can now use either the ```object_detection_tutorial.ipynb``` or ```test_pipe_aws.py``` file to test your images.
