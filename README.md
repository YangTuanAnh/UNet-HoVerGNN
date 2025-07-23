# UNet-HoVerGNN: Structured Graph Integration into HoVerNet for Enhanced Nuclei Segmentation and Classification

![Framework Overview](diagram.png)

---

## Abstract

We present **UNet-HoVerGNN**, a unified framework that enhances nuclei instance segmentation and classification by combining convolutional visual encoding with graph-based reasoning. Built on a U-Net backbone with a ResNet-34 encoder, our model extracts rich visual features, constructs \$k\$-nearest neighbor graphs over detected nuclei, and applies a graph neural network to model inter-nuclear relationships. Node and edge features are extracted via bilinear sampling and positional encodings from the shared encoder. Evaluated on three histopathology benchmarks—MoNuSAC, PanNuke, and CoNSeP—our approach consistently outperforms or matches state-of-the-art methods in Aggregated Jaccard Index (AJI), F1-score, and Panoptic Quality (PQ). The code and trained models are available at:
**[GitHub Repository](https://github.com/YangTuanAnh/UNet-HoVerGNN)**
**[Dataset on Kaggle](https://www.kaggle.com/datasets/yangtunanh/monusac-pannuke-consep/data)**
**[Models on Kaggle](https://www.kaggle.com/models/yangtunanh/hovergnn-monusac/)**

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

