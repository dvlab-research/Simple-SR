# Simple-SR

The repository includes MuCAN, LAPAR, etc. The testing code is first released. We will gradually add the training part and more methods.

### MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution (ECCV 2020)
 
Wenbo Li, Xin Tao, Taian Guo, Lu Qi, Jiangbo Lu, Jiaya Jia

[paper](https://arxiv.org/abs/2007.1180)

### LAPAR: Linearly-Assembled Pixel-Adaptive Regression Network for Single Image Super-resolution and Beyond (NeurIPS 2020)

Wenbo Li, Kun Zhou, Lu Qi, Nianjuan Jiang, Jiangbo Lu, Jiaya Jia


### Usage

1. Clone the repository
    ```shell
    git clone https://github.com/Jia-Research-Lab/Simple-SR.git
    ```
2. Install the dependencies
    - Python >= 3.7
    - PyTorch >= 1.2
    - spatial-correlation-sampler
    ```shell
    pip install spatial-correlation-sampler
    ```
    - Other packages
    ```shell
    pip install -r requirements.txt
    ```

3. Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1c-KUEPJl7pHs9btqHYoUJkcMPKViObgJ?usp=sharing)
    - MuCAN\_REDS.pth: trained on REDS dataset
    - MuCAN\_Vimeo90K: trained on Vimeo90K dataset

4. Test
    python3 test_sample.py --model_path /model/path --input_path /input/path --output_path /output/path


### Bibtex
    ```shell
    @article{li2020mucan,
          title={MuCAN: Multi-Correspondence Aggregation Network for Video Super-Resolution},
            author={Li, Wenbo and Tao, Xin and Guo, Taian and Qi, Lu and Lu, Jiangbo and Jia, Jiaya},
              journal={arXiv preprint arXiv:2007.11803},
                year={2020}
    }
    ```
