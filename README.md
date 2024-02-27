# IDExpO

Yuya Yoshikawa, Tomoharu Iwata, **Explanation-Based Training with Differentiable Insertion/Deletion Metric-Aware Regularizers,** The 27th International Conference on Artificial Intelligence and Statistics (AISTATS2024), Valencia, Spain, May 2024.

## Installation
Package `idexpo` can be installed in this repository by the following command:
```
pip install .
```

## Usage Examples
### Training and Evaluation

First, please download ResNet-18 pre-trained weights trained on CIFAR-10 (resnet18_cifar10.pth) from [Google Drive](https://drive.google.com/drive/folders/1-cjR618ECIslf2kPcLlmudyuFStRqvw_?usp=drive_link), and place it into `examples/weights` directory.
Then, you can fine-tune ResNet-18 with ID-ExpO and evaluate the fine-tuned model. 
```
cd examples
python train_cifar10_gradcam.py
```

## License
Please see [LICENSE.txt](./LICENSE.txt).


## TODO
- [x] Add implementation for Grad-CAM
- [ ] Add implementation for LIME
- [ ] Add implementation for tabular data
