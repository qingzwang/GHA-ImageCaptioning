# GHA-ImageCaptioning

This is the code for our papers: "Gated Hierarchical Attention for Image Captioning" and [CNN+CNN: Convolutional Decoders for Image Captioning](https://arxiv.org/abs/1805.09019). To run it you should first install [Pytorch 0.3.0](https://pytorch.org/docs/0.3.0/).

## Train
1. Download the MSCOCO2014 dataset [here](http://cocodataset.org/#download).
2. Unzip the files, and you put the training and validation images in the same folder. If you want to preprocess the dataset, please use **ak_build_vocab.py** in the **data** folder.
3. Let *self.image_dir* in **train.py** equal to the path of the folder in step 2. Also, you can change other parameters in the configuration.
## Inference
After training you can use the **inference.py** to generate captions for the images in the test split. Also, you should assign the path of the image folder to *self.image_dir*.
