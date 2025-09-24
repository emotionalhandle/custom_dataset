# Custom Dataset Workflow

A comprehensive machine learning workflow for processing image datasets with Hydrus integration, JoyTag embeddings, and series classification.

## Overview

This repository contains a complete workflow for:
1. **Dataset Mapping**: Extract image hashes from Hydrus database
2. **JoyTag Processing**: Generate embeddings using JoyTag model
3. **Series Classification**: Train classifiers on series tags
4. **Web Interface**: Browse and manage predictions

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Hydrus DB     │───▶│  Dataset Mapper  │───▶│  JoyTag Model   │
│   (Images)      │    │  (Extract Hash)   │    │  (Embeddings)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Series Tags     │    │  Classifier     │
                       │  (Training)      │    │  (Training)     │
                       └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────────────────────────────┐
                       │           Web Interface                 │
                       │      (Browse & Manage)                 │
                       └─────────────────────────────────────────┘
```

## Components

### Core Scripts

- **`dataset_hardcore_mapper.py`**: Maps Hydrus dataset structure and extracts image hashes
- **`advanced_joytag_processing.py`**: Processes images with JoyTag model to generate embeddings
- **`train_series_classifier_images.py`**: Trains series classification models

### JoyTag Integration

- **`joytag/Models.py`**: Core JoyTag model implementation
- **`joytag/config.json`**: Model configuration
- **`joytag/requirements.txt`**: JoyTag dependencies

### Web Interface

- **`webapp/app.py`**: Flask web application
- **`webapp/templates/index.html`**: Main web interface
- **`webapp/requirements.txt`**: Web app dependencies

### Configuration

- **`config.py`**: Hydrus API configuration
- **`requirements_iafd.txt`**: Additional dependencies

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Hydrus client with API access
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/emotionalhandle/custom_dataset.git
   cd custom_dataset
   ```

2. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install torch torchvision torchaudio
   pip install transformers einops safetensors pillow
   pip install requests beautifulsoup4 lxml
   pip install scikit-learn matplotlib seaborn
   
   # Web interface
   cd webapp
   pip install -r requirements.txt
   cd ..
   ```

3. **Download JoyTag model**:
   ```bash
   # Download from HuggingFace (model.safetensors file)
   # Place in joytag/ directory
   ```

4. **Configure Hydrus API**:
   ```python
   # Edit config.py with your Hydrus API settings
   API_URL = "http://127.0.0.1:45869"
   API_KEY = "your_api_key_here"
   ```

## Usage

### 1. Dataset Mapping

Extract image hashes from your Hydrus database:

```bash
python dataset_hardcore_mapper.py
```

This creates:
- `mappings/dataset_mapping_<timestamp>.json`: Complete dataset structure
- `mappings/dataset_hashes_<timestamp>.txt`: List of all image hashes

### 2. JoyTag Processing

Generate embeddings for your images:

```bash
python advanced_joytag_processing.py [batch_size] [max_hashes]
```

Options:
- `batch_size`: Number of images to process at once (default: 20)
- `max_hashes`: Maximum number of hashes to process (default: 0 = all)
- `--sample=1000`: Process only 1000 random images
- `--no-cache`: Disable caching
- `--force-regenerate`: Regenerate all embeddings

Outputs:
- `advanced_embeddings/`: Individual embedding files (`.pt`)
- `advanced_joytag_complete_<timestamp>.json`: Complete processing results

### 3. Series Classification Training

Train a series classifier on your data:

```bash
python train_series_classifier_images.py [dataset_file] \
    --frozen-epochs 3 \
    --unfrozen-epochs 10 \
    --unfreeze-last-k 1 \
    --batch-size 32 \
    --lr-head 1e-3 \
    --lr-backbone 1e-4 \
    --balance-method weighted \
    --output-dir series_classifier_output_images
```

### 4. Web Interface

Launch the web interface to browse results:

```bash
cd webapp
python app.py
```

Access at: `http://localhost:5000`

## Workflow Examples

### Basic Processing Pipeline

```bash
# 1. Map your dataset
python dataset_hardcore_mapper.py

# 2. Process with JoyTag (sample 1000 images)
python advanced_joytag_processing.py 20 0 --sample=1000

# 3. Train classifier
python train_series_classifier_images.py \
    --frozen-epochs 3 \
    --unfrozen-epochs 10 \
    --batch-size 32

# 4. Launch web interface
cd webapp && python app.py
```

### Advanced Processing

```bash
# Process with custom settings
python advanced_joytag_processing.py 50 5000 \
    --sample-frac=0.1 \
    --sample-seed=42 \
    --recompute-tags

# Train with specific configuration
python train_series_classifier_images.py \
    --frozen-epochs 5 \
    --unfrozen-epochs 15 \
    --unfreeze-last-k 2 \
    --balance-method focal \
    --focal-gamma 2.0 \
    --focal-alpha 0.25
```

## File Structure

```
custom_dataset/
├── dataset_hardcore_mapper.py      # Dataset mapping script
├── advanced_joytag_processing.py  # JoyTag processing script
├── train_series_classifier_images.py  # Training script
├── config.py                      # Configuration
├── requirements_iafd.txt          # Dependencies
├── joytag/                        # JoyTag model files
│   ├── Models.py
│   ├── config.json
│   ├── requirements.txt
│   └── README.md
├── webapp/                        # Web interface
│   ├── app.py
│   ├── requirements.txt
│   └── templates/
│       └── index.html
├── mappings/                      # Generated mappings
├── advanced_embeddings/           # Generated embeddings
└── series_classifier_output_images/  # Training outputs
```

## Configuration

### Hydrus API Settings

Edit `config.py`:
```python
API_URL = "http://127.0.0.1:45869"  # Your Hydrus API URL
API_KEY = "your_api_key_here"        # Your API key
REQUEST_TIMEOUT = 30                 # Request timeout
```

### JoyTag Model

The JoyTag model files should be placed in the `joytag/` directory:
- `model.safetensors`: Main model weights (download from HuggingFace)
- `config.json`: Model configuration (included)
- `Models.py`: Model implementation (included)

## Output Files

### Dataset Mapping
- `mappings/dataset_mapping_<timestamp>.json`: Complete dataset structure
- `mappings/dataset_hashes_<timestamp>.txt`: Hash list

### JoyTag Processing
- `advanced_embeddings/<hash>.pt`: Individual embedding files
- `advanced_joytag_complete_<timestamp>.json`: Processing results
- `cache/advanced_joytag_cache.json`: Processing cache

### Training Outputs
- `series_classifier_output_images/best_model.pth`: Trained model
- `series_classifier_output_images/results.json`: Training results
- `series_classifier_output_images/training_history.png`: Training curves
- `series_classifier_output_images/class_metrics.png`: Class performance

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in processing scripts
2. **Hydrus API connection failed**: Check API URL and key in `config.py`
3. **Model loading errors**: Ensure JoyTag model files are in correct location
4. **Web interface not loading**: Check Flask dependencies and port availability

### Performance Tips

- Use GPU acceleration for faster processing
- Adjust batch sizes based on available memory
- Use sampling for large datasets during development
- Enable caching for repeated processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **JoyTag**: For the excellent vision model
- **Hydrus**: For the powerful media management system
- **PyTorch**: For the deep learning framework
- **Flask**: For the web framework