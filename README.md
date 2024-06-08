# Structural Analysis of Asian and African Rice Panicles via Transfer Learning

## Conference: [**APSIPA ASC 2024**](http://www.apsipa2024.org/index.html)

## Repo Collaborators (Not Paper Authors)

**Lam Thai Nguyen** and **Trung Kien Pham**.

## Status

**Ongoing**

## Directory Tree

```
.
├── data/
│   ├── metadata
│   ├── original_images/
│   │   ├── O. glaberrima
│   │   └── O. sativa
│   ├── original_ricepr/
│   │   ├── O. glaberrima
│   │   └── O. sativa
│   ├── junction_detection_result/
│   │   └── {T,K}_{1,2,3}_UNET.xlsx
│   └── segmentation_result/
│       ├── {K,T,TK}_{1,2,3}_{1,2,3,4,5}.xlsx
│       └── metadata.xlsx
├── src/
│   ├── pipeline.py
│   ├── image_processor/
│   │   ├── RicePanicle.py
│   │   ├── thinning.py
│   │   ├── skeletonize.py
│   │   ├── clustering.py
│   │   ├── evaluation.py
│   │   └── AccuracyManager.py
│   ├── examples/
│   │   ├── image_annotation_segmentation.ipynb
│   │   └── image_processing_pipeline.ipynb
│   └── utils/
│       ├── evaluation_image_generating.py
│       ├── ricepr_manipulate.py
│       ├── json2binary.py
│       ├── bounding_boxes.py
│       └── model_selection.py
├── images/
│   ├── pipeline/
│   │   ├── pipeline/
│   │   │   └── {T,K}_{1,2,3}/
│   │   │       └── {20,40} images
│   │   ├── merge_pred/
│   │   │   └── {T,K}_{1,2,3}/
│   │   │       └── {20,40} images  
│   │   ├── junction_image
│   │   ├── junction_raw_image
│   │   └── skeleton_image
│   ├── model_predictions/
│   │   └── {T,K}_{1,2,3}_{1,2,3,4,5}_UNET_UNET/
│   │       └── {20,40} images
│   ├── annotated/
│   │   ├── annotated-K
│   │   └── annotated-T
│   ├── bounding_boxes/
│   │   ├── grains_junctions
│   │   └── primary_branches
│   └── raw_images_512
└── README.md
```

## Dir Description

- **data/**: Raw data and its information and text-format results.
- **src/**: Examples and Code for image processing.
- **images/**: Processed images ~ image-format results. 

## Examples

- Useful code examples are provided [HERE](src/examples).

## Call For Paper

![Alt text](<APSIPA ASC 2024/CFP/CFP.png>)