# VariTex: Variational Neural Face Textures

[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![Python 3.6](https://img.shields.io/badge/python-3.8.5-green.svg)

![Teaser](https://ait.ethz.ch/people/buehler/public/varitex/teaser.png)

This is the official repository of the paper:

> **VariTex: Variational Neural Face Textures**<br>
> [Marcel C. Bühler](https://ait.ethz.ch/people/buehler), [Abhimitra Meka](https://www.meka.page/), [Gengyan Li](https://ait.ethz.ch/people/lig/), [Thabo Beeler](https://thabobeeler.com/), and [Otmar Hilliges](https://ait.ethz.ch/people/hilliges/).<br>
> **Abstract:** *Deep generative models have recently demonstrated the ability to synthesize photorealistic images of human faces with novel identities. A key challenge to the wide applicability of such techniques is to provide independent control over semantically meaningful parameters: appearance, head pose, face shape, and facial expressions. In this paper, we propose VariTex - to the best of our knowledge the first method that learns a variational latent feature space of neural face textures, which allows sampling of novel identities. We combine this generative model with a parametric face model and gain explicit control over head pose and facial expressions. To generate images of complete human heads, we propose an additive decoder that generates plausible additional details such as hair. A novel training scheme enforces a pose independent latent space and in consequence, allows learning of a one-to-many mapping between latent codes and pose-conditioned exterior regions. The resulting method can generate geometrically consistent images of novel identities allowing fine-grained control over head pose, face shape, and facial expressions, facilitating a broad range of downstream tasks, like sampling novel identities, re-posing, expression transfer, and more.*


# Code and Models

## Code, Environment
- [ ] Clone repository: `git clone https://github.com/mcbuehler/VariTex.git` 
- [ ] Create environment: `conda env create -f environment.yml` and activate it `conda activate varitex`. 


## Data 
We train on the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) and we use the [Basel Face Model 2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) (BFM). Please download the following:

- [ ] FFHQ: Follow the instructions in the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset) to obtain the images (.png). Download "Aligned and cropped images at 1024×1024".
- [ ] Preprocessed dataset: [Download](https://ait.ethz.ch/people/buehler/public/varitex/preprocessed_dataset.zip) (~15 GB) and unzip.
- [ ] Basel Face Model: Request the [model](https://faces.dmi.unibas.ch/bfm/bfm2017.html) ("model2017-1_face12_nomouth.h5"") and download the [UV parameterization](https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json).
- [ ] Pretrained models: [Download](https://ait.ethz.ch/people/buehler/public/varitex/pretrained.zip) and unzip.
- [ ] Move the downloaded files to the correct locations (see below)

Environment variables should point to your data, facemodel, and (optional) output folder: `export DP=<YOUR_DATA_FOLDER>; export FP=<YOUR_FACEMODEL_FOLDER>; export OP=<YOUR_OUTPUT_FOLDER>`.
We assume the following folder structure. 
* `$DP/FFHQ/images`: Folder with *.png files from FFHQ
* `$DP/FFHQ/preprocessed_dataset`: Folder with the preprocessed datasets. Should contain .npy files "R", "t", "s", "sp", "ep", "segmentation", "uv", "filename", and a .npz file "dataset_splits".
* `$FP/basel_facemodel/`: Folder where the BFM model files are located. Should contain "model2017-1_face12_nomouth.h5" and "face12.json".
 
## Using the Pretrained Model

Make sure you have downloaded the pretrained model (link above).
Define the checkpoint file: `export CP=<PATH_TO_CHECKPOINT>.ckpt`

#### Demo Notebook
 Run the notebook `CUDA_VISIBLE_DEVICES=0 jupyter notebook` and open `demo.ipynb`. 

#### Inference Script
The inference script runs three different modes on the FFHQ dataset:
1. Inference on the extracted geometries and original pose (`inference.inference_ffhq`)
2. Inference with extracted geometries and multiple poses (`inference.inference_posed_ffhq`)
3. Inference with random geometries and poses (`inference.inference_posed`)

You can adjust the number of samples with the parameter `n`.

`CUDA_VISIBLE_DEVICES=0 python varitex/inference.py --checkpoint $CP --dataset_split val`.


## Training
Run `CUDA_VISIBLE_DEVICES=0 python varitex/train.py`.

If you wish, you can set a variety of input parameters. Please see `varitex.options`.

A GPU with 24 GB VMem should support batch size 7. If your GPU has only 12 GB, please use a lower batch size.

Training should converge after 44 epochs, which takes roughly 72 hours on a NVIDIA Quadro RTX 6000/8000 GPU.

## Implementation Details

The VariTex architecture consists of several components (in `varitex/modules`). We pass on a dictionary from one component to the next. The following table lists the classes / methods with their corresponding added tensors.


| Class / Method | Adds... |
|---	|---
| varitex.data.hdf_dataset.NPYDataset | IMAGE_IN, IMAGE_IN_ENCODE, SEGMENTATION_MASK, UV_RENDERED |	
| varitex.modules.encoder.Encoder  | IMAGE_ENCODED |
| varitex.modules.generator.Generator.forward_encoded2latent_distribution | STYLE_LATENT_MU, STYLE_LATENT_STD |
| varitex.modules.generator.Generator.forward_sample_style | STYLE_LATENT |
| varitex.modules.generator.Generator.forward_latent2featureimage | LATENT_INTERIOR, LATENT_EXTERIOR |
| varitex.modules.decoder.Decoder | TEXTURE_PERSON |
| varitex.modules.generator.Generator.sample_texture | FACE_FEATUREIMAGE |
| varitex.modules.decoder.AdditiveDecoder | ADDITIVE_FEATUREIMAGE |
| varitex.modules.generator.Generator.forward_merge_textures | FULL_FEATUREIMAGE |
| varitex.modules.feature2image.Feature2ImageRenderer | IMAGE_OUT, SEGMENTATION_PREDICTED |

## Acknowledgements
We implement our pipeline in [Lightning](https://www.pytorchlightning.ai/) and use the [SPADE](https://github.com/NVlabs/SPADE) discriminator. The neural rendering is inspired by [Neural Voice Puppetry](https://github.com/keetsky/NeuralVoicePuppetry). We found the [pytorch3d](https://github.com/facebookresearch/pytorch3d) renderer very helpful.


## License
Copyright belongs to the authors.
All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

