# UNet-HoVerGNN: Structured Graph Integration into HoVerNet for Enhanced Nuclei Segmentation and Classification

![Framework Overview](diagram.png)

---

## Abstract

We propose **UNet-HoVerGNN**, a hybrid framework for nuclei instance segmentation and classification that combines CNN-based segmentation with graph-based reasoning. Nuclei centroids are used to construct $k$-nearest neighbor graphs, where node and edge features are extracted from a pretrained U-Net encoder. A GNN refines classification by capturing spatial and contextual dependencies. Evaluated on MoNuSAC, PanNuke, and CoNSeP, our model achieves strong results: 0.603 AJI, 0.641 F1, 0.622 PQ on MoNuSAC; 0.644 AJI, 0.682 F1, 0.661 PQ on PanNuke; and 0.585 AJI, 0.608 F1, 0.595 PQ on CoNSeP, outperforming or matching state-of-the-art models like HoVerNet and similar graph-based methods. Code is available at: **[GitHub Repository](https://github.com/YangTuanAnh/UNet-HoVerGNN)**
**[Dataset on Kaggle](https://www.kaggle.com/datasets/yangtunanh/monusac-pannuke-consep/data)**
**[Models on Kaggle](https://www.kaggle.com/models/yangtunanh/hovergnn-monusac/)**
**[Reproduced experiments](https://www.kaggle.com/code/yangtunanh/unet-hovergnn-reproducibility)**

---

## Dataset Setup

To run the training and evaluation notebooks:

* Download the dataset from Kaggle and place it in a local folder.
* Redirect the `DATA_PATH` variable in the notebook to point to the dataset location.
* Ensure required dependencies (PyTorch, PyTorch Geometric, etc.) are installed.

Supported datasets:

* **MoNuSAC**: Multi-organ nuclei segmentation and classification.
* **PanNuke**: Pan-cancer nuclei classification.
* **CoNSeP**: Colorectal cancer nuclei segmentation.

---
## Training and Evaluation

To train and evaluate the UNet-HoVerGNN model, navigate to the src directory and run the main.py script with the desired dataset and number of classes. Command-line arguments --dataset and --num_classes override the default values specified in config.py.

Example usage:

```sh
cd src
python -m venv .venv
source .venv/bin/activate
# ".venv/Scripts/activate" (Windows)
pip install -r requirements.txt

python main.py --dataset MoNuSAC --num_classes 5 \
               --data_path /path/to/data \
               --output_path /path/to/output
               
python main.py --dataset PanNuke --num_classes 6 \
               --data_path /path/to/data \
               --output_path /path/to/output

python main.py --dataset CoNSeP_Tiled --num_classes 8 \
               --data_path /path/to/data \
               --output_path /path/to/output
```
---

## Performance Comparison

| **Model**           |     **AJI** |    **F1** |    **PQ** |     **AJI** |    **F1** |    **PQ** |    **AJI** |    **F1** |    **PQ** |
| ------------------- | ----------: | --------: | --------: | ----------: | --------: | --------: | ---------: | --------: | --------: |
|                     | **MoNuSAC** |           |           | **PanNuke** |           |           | **CoNSeP** |           |           |
| HoVerNet            |       0.599 |     0.613 |     0.604 |   **0.653** |     0.621 |     0.503 |      0.544 |     0.510 |     0.549 |
| Mask2Former         |       0.577 |     0.568 |     0.559 |       0.616 |     0.666 |     0.480 |      0.464 |     0.482 |     0.414 |
| TSFD-net            |       0.461 |     0.499 |     0.350 |       0.621 |     0.513 |     0.413 |      0.458 |     0.415 |     0.439 |
| SENC (HoVerNet)     |       0.599 |     0.613 | **0.692** |   **0.653** |     0.621 |     0.528 |      0.544 |     0.510 | **0.595** |
| **Ours (Pretrain)** |       0.230 |     0.271 |     0.248 |   **0.653** | **0.695** | **0.672** |      0.484 |     0.498 |     0.490 |
| **Ours (Finetune)** |   **0.603** | **0.641** |     0.622 |       0.644 |     0.682 |     0.661 |  **0.585** | **0.608** | **0.595** |

