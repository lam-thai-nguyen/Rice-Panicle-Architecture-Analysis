# Architecture Analysis of Rice Panicle using Deep Learning

Collaborators: **Lam Thai Nguyen** and **Trung Kien Pham**

## Folder/File/Script description<a id="description"></a>

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
- **utils**: tools/utils. <[HOW TO USE UTILS](#basic-operations)>
- **results**: for reference purposes, knowing what each procedure produces.

## Instructions - Basic Operations<a id="basic-operations"></a>

- **STEP 1**: Create an Object `rice_panicle = RicePanicle(user, file_path)`
  - user={T, K}  # T stands for Thai, K stands for Kien.
  - file_path  # This can be either json path or original path of your rice panicle.
- **STEP 2**: Methods we can use are listed here
  - `rice_panicle.return_info() -> None` shows you the information of your rice panicle like name, species, ...
  - `rice_panicle.json2binary() -> None` turns a json file containing the annotation result to a binary image under the same name.
  - `rice_panicle.generate_bbox_grains_junctions() -> None` does what the name says.
  - `rice_panicle.generate_bbox_pb() -> None` does what the name says.
  - `thin(self, method: str, _pre_process: bool, _post_process: bool, **kwargs) -> np.ndarray` uses specified method *{zhang, gradient}* to thin/skeletonize the binary image along with *pre-processing and post-processing*.
  - `cluster(self, method: str) -> np.ndarray` finds the junctions in the skeleton image. Will take no effect when skeleton is not created yet.
  - `thin_cluster(self, thin_method: str, _pre_process: bool, _post_process: bool, cluster_method: str, **kwargs) -> list[np.ndarray]`: a mix between thin() and cluster().
  - `rice_panicle.imshow_binary() -> None` -> creates a figure of the processed binary image. Will only take effect IF 1) is created and 2) is processed.
  - `rice_panicle.imshow_skeleton() -> None` -> creates a figure of the processed skeleton image. Will only take effect IF 1) is created and 2) is processed.
- **STEP 3**: Undergoing...
- **NOTES**: All files are saved under the file paths suitable only for this  repository. To find those results, check the [DESCRIPTION](#description) again