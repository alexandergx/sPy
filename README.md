# sPy - python ML image classifier
Train a neural network to recognize and classify images with Python's TensorFlow library.

# How-To
Editing the initial user variables under tfic.py -

Place your images dataset for training under a directory and assign the "directory/" name to the image_dir variable.
Select some classification test samples to place under a "test_directory/" assigned to the test_dir variable.
Input how many images you want to train over.
Input an image size (default 128). The greater the size (e.g. 256, 512) the longer the training model will take but the greater the accuracy.
Input your class names (default: cat, dog, flower) of the object(s) your are classifying.

Select "create training data" to process your dataset and output a usable "default.npz" file.

Select "run training model" to train your model on your "default.npz" file and output a "default.model" file.

Select "classify images" to run your model on your test samples.
