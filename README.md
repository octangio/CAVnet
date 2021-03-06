# A deep learning network for classifying arteries and veins in montaged wide-field OCT angiograms   

[ [**Paper**](https://www.sciencedirect.com/science/article/pii/S2666914522000380)]

By [**Min Gao**](https://scholar.google.com/citations?user=T1vzVnYAAAAJ&hl=en), [**Yukun Guo**](https://scholar.google.com/citations?user=BCrQPWUAAAAJ&hl=en&oi=sra), [**Tristan T.Hormel**](https://scholar.google.com/citations?user=jdD1rGwAAAAJ&hl=en), [**Kotaro Tsuboi**](https://www.researchgate.net/profile/Kotaro-Tsuboi-2), [**George Pacheco**](https://www.linkedin.com/in/george-pacheco-bs-coa-32190a154), David Poole, [**Steven T. Bailey**](https://www.researchgate.net/profile/Steven-Bailey-10), [**Christina J. Flaxel**](https://orcid.org/0000-0001-9353-9862), [**David Huang**](https://scholar.google.com/citations?user=SqEvY68AAAAJ&hl=en), [**Thomas S. Hwang**](https://www.researchgate.net/profile/Thomas-Hwang-2), [**Yali Jia**](https://scholar.google.com/citations?user=hfBY5K8AAAAJ&hl=en&oi=sra)

This repo is the official implementation of "[**A deep learning network for classifying arteries and veins in montaged wide-field OCT angiograms**](https://www.ophthalmologyscience.org/article/S2666-9145(22)00038-0/fulltext)".

This software is copyrighted and may only be used for academic research.

Please cite this paper if you use any component of the software.

Gao M, Guo Y, Hormel TT, Tsuboi K, Pacheco G, Poole D, Bailey ST, Flaxel CJ, Huang D, Hwang TS, Jia Y. A deep learning network for classifying arteries and veins in montaged wide-field OCT angiograms. Ophthalmology Science. 2022 Apr 1:100149.

## Introduction

Artery and vein abnormalities in the retina are important biomarkers for disease diagnosis. Distinguishing arteries from veins allows us to identify variation in the way disease affects each. In this study, we propose a convolutional neural network that classifies arteries and veins (CAVnet) on montaged wide-field optical coherence tomographic angiography (OCTA) en face images. This method takes the OCTA images as input and outputs the segmentation results with arteries and veins identified. We not only classify arteries and veins down to the level of precapillary arterioles and postcapillary venules, but also detect the intersection of arteries (or arterioles) and veins (or venules). The results show CAVnet has high accuracy on differentiating arteries and veins in diabetic retinopathy (DR) and branch retinal vein occlusion (BRVO) cases. These classification results are robust across two instruments and multiple scan volume sizes. Measurements of arterial and venous caliber or tortuosity made using our algorithm???s output show differences between healthy and diseased eyes, indicating potential benefits for disease diagnosis.

#### Figure 1. Algorithm flowchart.

![img](./Figures/cavnet.jpg)
## Getting Start

### Clone our repo

```bash
git clone git@github.com:octangio/CAVnet.git
cd CAVnet
```

### install packages

  ```bash
  pip install -r requirements.txt
  ```
  The code version we used in the paper is CAVnet-1.0.
### Train model on your own data

- prepare data
  
  The en face angiograms have normalized the range of decorrelation value (SSADA) from (0.02, 0.3) to the range of (0, 255). The data set folder should be like the following structure.

    ```bash
    dataset
    |
    |-- train_image
    |   |
    |   |_list.txt
    |   |- image_0001.png
    |   |- image_0002.png
    |   |- ...
    |
    |-- train_label
    |   |
    |   |_list.txt
    |   |- label_0001.png
    |   |- label_0002.png
    |   |- ...
    |
    |-- valid_image
    |   |
    |   |_list.txt
    |   |- image_0001.png
    |   |- image_0002.png
    |   |- ...
    |
    `-- valid_label
        |
        |   |_list.txt
        |- label_0001.png
        |- label_0002.png
        |- ...
  ```
  Then you need to generate the path list of image and label. 

- Training
  
  ```bash
  python my_train.py --train_image=./dataset/train_image/_list.txt --train_label=./dataset/train_label/_list.txt --valid_image=./dataset/valid_image/_list.txt --valid_label=./dataset/valid_label/_list.txt --batch_size=2 --input_height=400 --input_width=400   
  ```
- Test

  ```bash
  python predict_results.py --test_data_path=./dataset/test_data_path --save_path=./dataset/AV_output --save_mat=./dataset/AV_out2mat --logdir=./logs/saved_model.hdf5
  ```
 #### Figure 2. Predicted results on healthy eyes and eyes with diabetic retinopathy.

![img](./Figures/results_DR.jpg)

 #### Figure 3. Predicted results on healthy eyes and eyes with branch retinal vein occlusion.

![img](./Figures/BRVO.jpg)

 #### Figure 4. Predicted results on scans with large-field-of view.

![img](./Figures/results_large_field_view.jpg)
