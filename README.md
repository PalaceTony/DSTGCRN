# Dynamic Spatial-Temporal Model for Carbon Emission Forecasting

This repository contains the implementation of the model Dynamic Spatial-Temporal Graph Convolutional Recurrent Network (DSTGCRN) presented in the manuscript "Dynamic Spatial-Temporal Model for Carbon Emission Forecasting".

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Model Training](#model-training)
- [Contact](#contact)

## Description

This project introduces a dynamic spatial-temporal modeling approach to forecast carbon emissions. The model leverages spatial correlations and temporal dynamics to provide more accurate predictions. The repository includes the source code and datasets.

## Requirements

The following packages are required to run the code provided in this repository:

- Python: `3.8.18`
- Hydra: `2.5`
- Hydra Core: `1.3.2`
- PyTorch: `2.0.1` or above
- NumPy: `1.24.4`

These are the core requirements; however, additional packages may be needed as you work through the project. Please install any other necessary packages as required.

## Model Training

To train the model, you will need to run the script located at `src/model/DSTGCRN/run.py`. By default, the script is configured to train on a dataset from China. The relative path `src/model/DSTGCRN/run.py` is based on the scenario where `DSTGCRN` is the folder you navigate to as the starting point.

You can also specify different datasets by using the `dataset` argument when running the script.

```bash
python src/model/DSTGCRN/run.py dataset=US
python src/model/DSTGCRN/run.py dataset=EU
```

## Contact

For any inquiries or further discussions, feel free to reach out at [mgong081@connect.hkust-gz.edu.cn](mailto:mgong081@connect.hkust-gz.edu.cn).

## Citation

If you find this work useful, please cite the following paper:

```
@article{gong2024dynamic,
  title={Dynamic spatial-temporal model for carbon emission forecasting},
  author={Gong, Mingze and Zhang, Yongqi and Li, Jia and Chen, Lei},
  journal={Journal of Cleaner Production},
  pages={142581},
  year={2024},
  publisher={Elsevier}
}
```

This paper is available at https://www.sciencedirect.com/science/article/pii/S0959652624020298
