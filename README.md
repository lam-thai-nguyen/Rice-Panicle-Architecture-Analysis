# Architecture Analysis of Rice Panicle using Deep Learning

## Collaborators

**Lam Thai Nguyen** and **Trung Kien Pham**.

## Directory Tree

```
.
├── data/
│   ├── metadata
│   ├── original_images
│   └── original_ricepr
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── image_processor/
│   │   ├── __init__.py
│   │   ├── RicePanicle.py
│   │   ├── thinning.py
│   │   ├── skeletonize.py
│   │   ├── clustering.py
│   │   └── evaluation.py
│   ├── examples/
│   │   ├── image_annotation_segmentation.ipynb
│   │   └── image_processing_pipeline.ipynb
│   └── utils/
│       ├── __init__.py
│       ├── evaluation_image_generating.py
│       ├── ricepr_manipulate.py
│       ├── json2binary.py
│       └── bounding_boxes.py
├── images/
│   ├── annotated
│   ├── bounding_boxes
│   ├── model_predictions/
│   │   ├── run_1
│   │   └── run_2
│   ├── pipeline/
│   │   ├── junction_image
│   │   ├── junction_raw_image
│   │   ├── merge_pred
│   │   └── skeleton_image
│   └── raw_images_512
└── README.md
```

## Dir Description

- **data/**: Raw data and its information.
- **src/**: Examples and Code for image processing.
- **images/**: Processed images ~ results. 

## Examples

- Useful code examples are provided [HERE](src/examples).