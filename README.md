# InPainTor🎨 : Context-Aware Segmentation and Inpainting in Real-Time

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Conda](https://img.shields.io/badge/conda-environment-green.svg)](https://docs.conda.io/en/latest/)

---
<p align="center">
  <a href="https://ciic.ipleiria.pt/">
    <img src="assets/CIIC_FCT_logo.png" width="700px" alt="CIIC CT Logo"/>
  </a>
</p>

---

**InPainTor**🎨 is a deep learning model designed for context-aware segmentation and inpainting in real-time. It
recognizes objects of interest and performs inpainting on specific classes while preserving the surrounding context.

<center>
  <img src="assets/training_gif.gif" alt="Training">
</center>

## 🚀 Features

- Real-time object recognition and inpainting
- Selective removal and filling of missing or unwanted objects
- Context preservation during inpainting
- Two-stage training process: segmentation and inpainting
- Support for COCO and RORD datasets

## 🚧 **WIP** (Work In Progress)

This project is currently under development. Use with caution and expect changes.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/InPainTor.git
   cd InPainTor
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate inpaintor
   ```

## 🖥️ Usage

### Training

To train the InPainTor model:

```bash
python src/train.py --coco_data_dir <path_to_COCO> --rord_data_dir <path_to_RORD> --seg_epochs <num_epochs> --inpaint_epochs <num_epochs>
```

<details>
<summary>Click to view all training arguments</summary>

- `--coco_data_dir`: Path to the COCO 2017 dataset directory (default: '/media/tiagociiic/easystore/COCO_dataset')
- `--rord_data_dir`: Path to the RORD dataset directory (default: '/media/tiagociiic/easystore/RORD_dataset')
- `--seg_epochs`: Number of epochs for segmentation training (default: 10)
- `--inpaint_epochs`: Number of epochs for inpainting training (default: 10)
- `--batch_size`: Batch size for training (default: 2)
- `--learning_rate`: Learning rate for the optimizer (default: 0.1)
- `--image_size`: Size of the input images, assumed to be square (default: 512)
- `--mask_size`: Size of the masks, assumed to be square (default: 256)
- `--model_name`: Name of the model (default: 'InPainTor')
- `--log_interval`: Log interval for training (default: 1000)
- `--resume_checkpoint`: Path to the checkpoint to resume training from (default: None)
- `--selected_classes`: List of class IDs for inpainting (default: [1, 72, 73, 77])

</details>

### Inference

To perform inference using the trained InPainTor model:

```bash
python src/inference.py --model_path "path/to/model.pth" --data_dir "path/to/data" --image_size 512 --mask_size 256 --batch_size 1 --output_dir "path/to/outputs"
```

## 📁 Project Structure

<details>
<summary>Click to view the repository structure</summary>

```plaintext
InpainTor/ 
├── assets/                   📂: Repository assets (images, logos, etc.)
├── checkpoints/              💾: Model checkpoints
├── logs/                     📃: Log files
├── notebooks/                📓: Jupyter notebooks
├── outputs/                  📺: Output files generated during inference, training and debugging
├── src/                      📜: Source code files
│   ├── __init__.py           📊: Initialization file
│   ├── data_augmentation.py  📑: Data augmentation operations
│   ├── dataset.py            📊: Dataset loading and preprocessing
│   ├── debug_model.py        📊: Model debugging
│   ├── inference.py          📊: Inference script
│   ├── layers.py             📊: Model layers
│   ├── losses.py             📊: Loss functions
│   ├── model.py              📑: InpainTor model implementation
│   ├── train.py              📊: Training script
│   └── visualizations.py     📊: Visualization functions
├── .gitignore                🚫: Files to ignore in Git
├── environment.yml           🎛️: Conda environment configuration
├── README.md                 📖: Project README file
└── main.py                   📜: Entry point for the InpainTor model
```

</details>

## 🧠 Model Architecture

The InPainTor model consists of three main components:

1. **SharedEncoder**: Encodes input images into a series of feature maps.
2. **SegmentorDecoder**: Decodes encoded features into segmentation masks.
3. **GenerativeDecoder**: Uses segmentation information to generate inpainted images.

The model is designed to be flexible, allowing for freezing and unfreezing of specific parts during training.

**Overall Model Architecture**
<center>
  <img src="assets/model_full.jpeg" height="300px" style="border: 1px solid gray" alt="Model">
</center>

**Model Components in Detail**
<center>
  <img src="assets/model_components.jpeg" width="700px" style="border: 1px solid gray" alt="Model">
</center>

**Model Training Process**
<center>
  <img src="assets/plot_training_log_full.png" width="700px" style="border: 1px solid gray" alt="Model">
</center>

## 📊 Dataset Requirements

<details>
<summary>RORD Inpainting Dataset Structure</summary>

The [RORD dataset](https://github.com/Forty-lock/RORD) should be organized as follows:

```
root_dir/
├── train/
│   ├── img/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── gt/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── img/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── gt/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

</details>

<details>
<summary>COCO Segmentation Dataset Structure</summary>

The [COCO dataset](https://cocodataset.org/#home) (2017 version with 91 classes) should be organized as follows:

```
root_dir/
├── train/
│   ├── img/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── gt/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── img/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── gt/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

For more information on COCO dataset classes, refer
to [this link](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/).

</details>

## 🤝 Contributing

Contributions to the InPainTor project are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes
4. Push to your fork and submit a pull request

We appreciate your contributions to improve InPainTor!

## 🙏 Acknowledgements

This work is funded by FCT - Fundação para a Ciência e a Tecnologia, I.P., through project with reference
2022.09235.PTDC.

## 📄 License

This project is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html).

---

For more information or support, please open an issue in the GitHub repository.