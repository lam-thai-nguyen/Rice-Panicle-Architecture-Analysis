# Architecture Analysis of Rice Panicle using Deep Learning

Collaborators: **Lam Thai Nguyen** and **Trung Kien Pham**

## Folder/File/Script description

- **dataset**
  - **metadata**:
    - **InfoAccession.xlsx**: Accession information 
    - **dataset_info.xlsx**: information about each rice panicle image
  - **original**: 200 rice panicle images (100 O. sativa and 100 O. glaberrima).
  - **original_512x512**: 200 rice panicle images resized to (512, 512).
  - **vertex_coordinates**: rice panicle's vertices coordinates (200 corresponding .ricepr files).
  - **annotated**: 2 versions of 200 x (binary image, json file).
    - **annotated-T**: annotated by **Lam Thai Nguyen**
    - **annotated-K**: annotated by **Trung Kien Pham**
  - **bbox**: 
    - **grains_junction**s: bounding boxes results for grains/junctions detection.
    - **primary_branches**: bounding boxes results for primary branches (Pb) detection.
- **utils**: tools/utils.