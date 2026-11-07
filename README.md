# Intelibin-Intelligent-Waste-Bin

A repository for Intelibin, an intelligent waste bin. Currently, it contains the dataset and depth-map generation script used in our experiments. The rest of the code will be added later.

## Intelibin Dataset

The dataset includes the RGB training, validation, and test sets used in our Intelibin experiments.

The depth maps are not included. To generate them, run the Python script provided in this repository. It uses Depth Anything V2 to generate a depth map for each RGB image, keeping the same training, validation, and test splits.

### Dataset sources

We combined images from the following public datasets:

- [Garbage Dataset, version 8](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/versions/8), published by Suman Kunwar on Kaggle.
- [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification), published by CCHANG on Kaggle.

The original images belong to their respective owners. Our contribution here is the combined dataset preparation, the splits used in our experiments, and the script for generating the corresponding depth maps. Please also cite the original datasets when using this work.
