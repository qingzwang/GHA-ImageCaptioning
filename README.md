# GHA-ImageCaptioning

This is the code for our papers: [Gated Hierarchical Attention for Image Captioning](https://arxiv.org/abs/1810.12535) and [CNN+CNN: Convolutional Decoders for Image Captioning](https://arxiv.org/abs/1805.09019). To run it you should first install [Pytorch 0.3.0](https://pytorch.org/docs/0.3.0/).

## Train
1. Download the MSCOCO2014 dataset [here](http://cocodataset.org/#download).
2. Unzip the files, and you put the training and validation images in the same folder. Put **captions_val2014.json** file in the **annotation** folder.
3. Download Karpathy's split [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip), and put it in the folder **data/files/**, then run **ak_build_vocab.py** in the **data** folder to preprocess the dataset.
4. Download COCO evaluation metrics [here](https://github.com/tylin/coco-caption). Copy all files to **models/coco_eval**.
3. Let *self.image_dir* in **train.py** equal to the path of the folder in step 2. Also, you can change other parameters in the configuration.
## Inference
After training you can use the **inference.py** to generate captions for the images in the test split. Also, you should assign the path of the image folder to *self.image_dir*.
