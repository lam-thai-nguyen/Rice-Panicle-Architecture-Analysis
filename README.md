# Architecture Analysis of Rice Panicle using Deep Learning

## Collaborators

**Lam Thai Nguyen** and **Trung Kien Pham**.

## Project Directory Tree

```
.
├── crack_segmentation/
│   ├── models
│   └── transfer-learning-results/
│       ├── run_1
│       └── run_2
├── data/
│   ├── metadata
│   ├── original_images
│   └── original_ricepr
├── src/
│   ├── image_processor/
│   │   ├── RicePanicle.py
│   │   ├── thinning.py
│   │   ├── clustering.py
│   │   └── evaluation.py
│   ├── examples/
│   │   ├── image_annotation_segmentation.ipynb
│   │   └── image_processing_pipeline.ipynb
│   └── utils/
│       ├── json2binary.py
│       └── bounding_boxes.py
├── images/
│   ├── annotated
│   ├── binary_images
│   ├── bounding_boxes
│   ├── junction_images
│   ├── raw_images_512
│   └── skeleton_images
└── README.md
```

## Dir Description

- **data/**: The raw data and its information.
- **src/**: Examples and Code for image processing.
- **images/**: Processed images ~ results. 