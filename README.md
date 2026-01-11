# MobileLLM: Semantic-Enhanced LLM for Cellular Traffic Prediction

> **Under Review** at *IEEE Transactions on Mobile Computing (TMC)*

## Abstract

MobileLLM integrates numerical spatiotemporal signals with semantic context understanding for multimodal cellular traffic prediction. Through parameter-efficient adaptation of GPT-2 (<10% trainable parameters), the framework achieves 0.88%-32.62% MAE improvements over strong baselines across SMS, voice call, and Internet traffic on three real-world urban datasets.

## Key Contributions

- **Dual-Pathway Architecture**: Bridges numerical time-series modeling with semantic context via partially-frozen GPT-2 and cross-attention fusion
- **Semantic Encoding**: Systematic conversion of urban contexts (zones, temporal patterns, events) into structured prompts
- **Comprehensive Validation**: Milan, Trentino, and Shanghai datasets with consistent improvements over Time-LLM, UrbanGPT, and STAEformer


## Quick Start

### Installation
```bash
git clone https://github.com/Dreamzz5/MobileLLM.git
cd MobileLLM
pip install -r requirements.txt
```

### Training
```bash
# Milan dataset
python train.py -d MILAN -g 0

# Trentino dataset
python train.py -d TRENTO -g 0

# Shanghai dataset
python train.py -d SHANGHAI -g 0
```

### Data
Download datasets from [Google Drive](https://drive.google.com/drive/folders/1TvMI6cjiXDvWwUXm_Z8r3j8nMNA-RDWi) and place in `data/` directory.

## Repository Structure

```
MobileLLM/
├── mobilellm/              # Core implementation
│   ├── MobileLLM.py       # Model architecture
│   ├── data_prepare.py    # Data processing
│   └── metrics.py         # Evaluation
├── train.py               # Training script
├── MobileLLM.yaml         # Configuration
└── requirements.txt       # Dependencies
```

## License

This project is licensed under the MIT License.

