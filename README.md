# ResNet Image Classification

A comprehensive deep learning project implementing ResNet architectures for image classification. This project compares ResNet-50 using transfer learning in both PyTorch and TensorFlow, and ResNet-18 trained from scratch using PyTorch. Includes an interactive Streamlit GUI for inference.

## Project Overview

This project explores the ResNet architecture, which solved the vanishing gradient problem by introducing residual learning through skip connections. The goal is to build efficient deep image classifiers using different training strategies and frameworks.

### Key Features
- **Multiple Implementations**: ResNet-50 transfer learning (TensorFlow & PyTorch) and ResNet-18 from scratch
- **Framework Comparison**: Practical experience with both PyTorch and TensorFlow
- **Transfer Learning**: Leverages ImageNet pretrained weights for efficient training
- **Training from Scratch**: Understanding deep network optimization challenges
- **Streamlit GUI**: User-friendly interface for real-time image classification
- **Performance Analysis**: Comparison of different training approaches

## Dataset

Uses ImageNet dataset for training and evaluation. For faster experimentation, a reduced dataset is used during development. Models achieve >70% accuracy with transfer learning approaches.

## Architecture: ResNet

ResNet (Residual Networks) addresses the degradation problem in very deep networks through residual learning.

**The Degradation Problem**: When deeper networks started converging, accuracy would saturate and then degrade rapidly—not due to overfitting, but due to optimization difficulties.

**The Solution - Skip Connections**: Instead of learning the full mapping H(x), networks learn the residual F(x) = H(x) - x. The final output becomes F(x) + x.

**Key Components**:

**Skip Connections**: Shortcuts that allow data to flow directly from one layer to later layers. If a layer doesn't need to modify data, weights can simply be set to zero, enabling perfect gradient flow.

**Bottleneck Blocks** (in ResNet-50/101/152): Use 1×1 convolutions to reduce dimensionality, perform computation, then expand back. This reduces computational cost while maintaining representational capacity.

**Residual Blocks**: The basic building unit where multiple layers learn a residual function F(x), then add it back to the original input x.

**Batch Normalization**: Normalizes layer inputs to accelerate training and improve stability.

## Project Approach

### Development Strategy

1. **Foundation Building**: Started with ResNet-50 transfer learning in TensorFlow using ImageNet pretrained weights
2. **Framework Exploration**: Replicated ResNet-50 transfer learning in PyTorch to compare implementations
3. **From-Scratch Training**: Implemented ResNet-18 from scratch in PyTorch to understand deep network optimization
4. **Practical Adaptations**:
   - Reduced dataset size for feasible training time
   - Limited epochs due to computational constraints
   - Transfer learning provided 70%+ accuracy despite limitations
   - From-scratch training achieved ~20% accuracy, demonstrating difficulty of training deep networks without pretrained weights

### Why Transfer Learning Works

ImageNet pretrained weights provide strong initial feature representations that have already learned to detect edges, textures, and patterns. Fine-tuning these weights on new data dramatically reduces training time and data requirements while achieving better accuracy than training from scratch with limited resources.

## File Structure

- `models/` - All model implementations
  - `ResNet18_PyTorch.ipynb` - ResNet-18 trained from scratch (PyTorch)
  - `ResNet50_TF_TL.ipynb` - ResNet-50 transfer learning (TensorFlow/Keras)
  - `ResNet50_PyTorch_TL.ipynb` - ResNet-50 transfer learning (PyTorch)
- `gui/` - Streamlit GUI application
  - `app.py` - Interactive inference interface
- `docs/` - Documentation and reference materials
- `requirements.txt` - Project dependencies
- `README.md` - This file

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.5+
- TensorFlow 2.15+ (for TF implementation)
- Streamlit
- NumPy, Pandas, Matplotlib, Pillow

### Setup
```bash
# Clone repository
git clone https://github.com/mdarshanrshaikh/resnet-image-classification.git
cd resnet-image-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Jupyter Notebooks

Run the individual model implementations:

**PyTorch Transfer Learning:**
```bash
jupyter notebook models/ResNet50_PyTorch_TL.ipynb
```

**TensorFlow Transfer Learning:**
```bash
jupyter notebook models/ResNet50_TF_TL.ipynb
```

**PyTorch From Scratch:**
```bash
jupyter notebook models/ResNet18_PyTorch.ipynb
```

### GUI Inference

Run the Streamlit application for interactive predictions:
```bash
streamlit run gui/app.py
```

The GUI allows you to:
- Upload images for classification
- Get model predictions with confidence scores
- Visualize top-5 predictions
- Compare different model performances

## Key Learnings

1. **Skip Connections Enable Deep Learning**: Residual connections allow networks to be much deeper by providing gradient shortcuts, making optimization significantly easier.

2. **Transfer Learning Efficiency**: Pretrained ImageNet weights provide enormous benefits. Even with reduced training data and limited epochs, transfer learning achieved 70%+ accuracy while from-scratch training only reached ~20%.

3. **Degradation Problem**: Without residual connections, simply stacking layers doesn't guarantee better performance. Deeper plain networks can have higher training error than shallower ones due to optimization difficulties.

4. **Framework Differences**: PyTorch and TensorFlow have different abstractions and coding styles, but produce comparable results. Framework choice depends on workflow preference and deployment requirements.

5. **Computational Trade-offs**: Training ResNet-50 from scratch requires significant computational resources. Transfer learning enables practical deep learning with limited hardware.

6. **Batch Normalization Importance**: BN layers are crucial for training stability, enabling faster convergence and better final accuracy.

## Model Performance

**Transfer Learning Results (70%+ accuracy)**:
- ResNet-50 PyTorch TL: Best performance with fast convergence
- ResNet-50 TensorFlow TL: Comparable accuracy with different implementation

**From-Scratch Training (~20% accuracy)**:
- ResNet-18 from scratch: Demonstrates optimization challenges in very deep networks without pretrained initialization

Despite computational constraints limiting epochs and data size, transfer learning proved dramatically more effective than training from scratch.

## References

**Paper**: Deep Residual Learning for Image Recognition (He et al., 2015)

**Key Concepts**:
- ResNet solves vanishing gradient and degradation problems
- Skip connections enable 100+ layer networks
- Bottleneck design for computational efficiency
- Residual learning framework is general and applicable to various tasks

**Resources**:
- ResNet Explanation: https://www.youtube.com/watch?v=woEs7UCaITo
- Original Paper: Available in docs/

## Contact

Questions or feedback? Feel free to open an issue or reach out.
