# single-face-detection
Hello welcome to my face detection

its a python model which can detect faces in images with 1 face at most

# Requirements

Ubuntu      " It's only tested on Ubuntu, so it may not work on Windows "

GPU : Any GPU that is works with PyTorch

vram : 8gb if you want to train  ' we train it in rtx 2070 '

numpy : https://numpy.org/

PyTorch : https://pytorch.org/

torchvision : https://pypi.org/project/torchvision/

openCV : https://pypi.org/project/opencv-python/

# Preview  

![preview](https://user-images.githubusercontent.com/49597655/133093044-c80eaa14-124f-4dc5-b6d7-d06e593d39a8.gif)

# Train :

* Download my pretrained model : https://drive.google.com/file/d/1sq1eZhRyQya1G8Lsdb5aqfVwEg5FhMH0/view?usp=sharing

OR 

* Download the faces DataSet and put it in project directory:  http://vis-www.cs.umass.edu/fddb/

* Download the no faces DataSet : 

* Run the train.py file : `$ python3 train.py`

# Test 

* Test an image `$ python3 test_image.py [your image path]` 

* Test a video `$ python3 test_video.py [your video path]`
