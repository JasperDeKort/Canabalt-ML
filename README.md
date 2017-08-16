# Canabalt-ML
Using a neural network to play Canabalt ( a 1 button endless runner) from player examples.

requirements:
  - windows 
  - canabalt in windowed mode, 800x600 in top left corner of screen
  - python 3.6
  - numpy
  - pandas
  - opencv3
  - pywin32
  
depending on the version used you also need one of these:
  - sklearn
  - tensorflow
  

the neural network to play the game is built from data of me playing the game. the performance at this time is not great. A version using a convoluted neural network in tensorflow is in the works. and there are plans to rebuild this using reinforcement learning instead.

all of this was developed using anaconda.

included are multiple different model training files:

### train_tensorflow.py
uses a neural network built with matmul and add functions. first version using tensorflow. outdated but kept for record purposes

### train_tensorflow_2.py
uses a neural network built using the tf.nn module. second version, kept for record purposes and non convolutional works

### train_tensorflow_cnn.py
uses a convolutional neural network built using the tf.nn model. current most updated version. 

### train_sklearn.py
uses the sklearn MLPclassifier.
  
  
## todo list
- gather more training data for rare occurances in game (inside hallways, dropping parts, etc)
- build reinforcement learning version

## done
- detect game screen coordinates on startup
- improve death detection with smaller and shorter compare to saved death screen
- modify saving and data preprocessing to save and load 1 file per run, preventing large file size of training data
