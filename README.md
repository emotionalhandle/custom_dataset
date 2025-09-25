# Custom Dataset Workflow

A comprehensive workflow for processing, analyzing, and detecting errors in image datasets using JoyTag embeddings and machine learning.

## 🚀 Quick Start

1. **Map your dataset:**
   ```bash
   python dataset_hardcore_mapper.py
   ```

2. **Process with JoyTag:**
   ```bash
   python advanced_joytag_processing.py
   ```

3. **Train series classifier:**
   ```bash
   python train_series_classifier_images.py
   ```

4. **Detect tagging errors:**
   ```bash
   python run_error_detection.py
   ```

5. **Review errors in web interface:**
   ```bash
   python webapp/app.py
   # Visit http://localhost:5000/error-review
   ```

## 📁 Core Components

### Dataset Processing
- **`dataset_hardcore_mapper.py`** - Maps Hydrus pages to image hashes
- **`advanced_joytag_processing.py`** - Extracts JoyTag embeddings and metadata
- **`train_series_classifier_images.py`** - Trains multi-label series classifier

### Error Detection
- **`find_mistagged_images.py`** - Detects missing or incorrect tags
- **`find_near_duplicates.py`** - Finds near-duplicate images using embeddings
- **`validate_series_predictions.py`** - Validates series classifier predictions
- **`run_error_detection.py`** - Master script to run all error detection methods

### Web Interface
- **`webapp/app.py`** - Flask web application for image analysis
- **`webapp/templates/error_review.html`** - Interactive error review dashboard

## 🔍 Error Detection Strategies

### 1. **Automated Tagging Error Detection**
The `find_mistagged_images.py` script analyzes your processed data for common tagging issues:

- **Missing tags**: Images that should have certain tags based on content
- **Incorrect tags**: Tags that don't match the image content
- **Inconsistent tagging**: Similar images with different tag patterns
- **Tag contradictions**: Conflicting tags on the same image

**Usage:**
```bash
python find_mistagged_images.py
```

**Output:** `mistagged_images_report_<timestamp>.json`

### 2. **Visual Similarity-Based Error Detection**
The `find_near_duplicates.py` script uses JoyTag embeddings to find potential duplicates or mislabeled images:

- **Near-duplicate detection**: Images with high visual similarity
- **Tag comparison**: Identifies inconsistent tagging between similar images
- **Similarity analysis**: Computes cosine similarity between embeddings

**Usage:**
```bash
python find_near_duplicates.py
```

**Output:** `near_duplicates_report_<timestamp>.json`

### 3. **Series Classification Validation**
The `validate_series_predictions.py` script validates your series classifier predictions against existing tags:

- **Prediction accuracy**: Measures how well predictions match existing tags
- **Missing predictions**: Series that exist in tags but weren't predicted
- **Incorrect predictions**: Series that were predicted but don't exist in tags
- **Confidence analysis**: Analyzes prediction confidence levels

**Usage:**
```bash
python validate_series_predictions.py
```

**Output:** `series_validation_report_<timestamp>.json`

### 4. **Interactive Web Dashboard**
The web interface provides a user-friendly way to review and fix tagging errors:

- **Error overview**: Statistics and summary of all detected errors
- **Filtering**: Filter errors by type, confidence, or hash
- **Visual review**: View images alongside their tags and predictions
- **Manual fixes**: Mark errors as fixed or ignored

**Usage:**
```bash
python webapp/app.py
# Visit http://localhost:5000/error-review
```

## 🛠️ Configuration

### Hydrus API Settings
Update `config.py` with your Hydrus API details:

```python
API_URL = "http://localhost:45869"
API_KEY = "your_api_key_here"
```

### JoyTag Model
The JoyTag model files are in the `joytag/` directory:
- `Models.py` - Core model implementation
- `config.json` - Model configuration
- `requirements.txt` - Python dependencies

## 📊 Understanding Error Reports

### Mistagged Images Report
```json
{
  "summary": {
    "total_images_analyzed": 1000,
    "errors_found": 45,
    "missing_tags": 23,
    "incorrect_tags": 22
  },
  "errors": [
    {
      "hash": "abc123...",
      "issue": "missing_tags",
      "missing_tags": ["person:jane_doe", "series:example_series"],
      "confidence": 0.85
    }
  ]
}
```

### Near Duplicates Report
```json
{
  "summary": {
    "total_pairs_found": 15,
    "similarity_threshold": 0.95
  },
  "similar_pairs": [
    {
      "hash1": "abc123...",
      "hash2": "def456...",
      "similarity": 0.97,
      "analysis": {
        "series_tags_diff": {
          "only_in_1": ["series:example_series"],
          "only_in_2": []
        }
      }
    }
  ]
}
```

### Series Validation Report
```json
{
  "summary": {
    "total_predictions": 500,
    "correct_predictions": 420,
    "accuracy": 0.84,
    "missing_predictions_count": 30,
    "incorrect_predictions_count": 50
  }
}
```

## 🎯 Best Practices

### 1. **Regular Error Detection**
Run error detection after major dataset updates:
```bash
python run_error_detection.py
```

### 2. **Review High-Confidence Errors First**
Focus on errors with high confidence scores (≥0.8) as they're most likely to be real issues.

### 3. **Use Visual Similarity for Duplicates**
Check near-duplicate pairs for inconsistent tagging - this often reveals tagging errors.

### 4. **Validate Series Predictions**
Regularly check if your series classifier is making accurate predictions against existing tags.

### 5. **Manual Review**
Use the web dashboard for manual review of complex cases that automated detection might miss.

## 🔧 Troubleshooting

### Common Issues

**"No processed dataset found"**
- Run `advanced_joytag_processing.py` first to create the dataset

**"No embeddings found"**
- Ensure the `advanced_embeddings/` directory exists and contains `.pt` files
- Check that JoyTag processing completed successfully

**"API connection failed"**
- Verify your Hydrus API settings in `config.py`
- Ensure Hydrus is running and accessible

**"Model loading failed"**
- Check that JoyTag model files are present in the `joytag/` directory
- Verify all dependencies are installed: `pip install -r joytag/requirements.txt`

### Performance Tips

- **Limit embeddings**: Use `max_embeddings` parameter to limit processing for large datasets
- **Batch processing**: Process errors in batches to avoid memory issues
- **Caching**: Enable caching in the web interface for faster repeated access

## 📈 Next Steps

1. **Fix detected errors** using the web dashboard
2. **Re-run error detection** to verify fixes
3. **Monitor error trends** over time
4. **Improve tagging consistency** based on findings
5. **Update series classifier** with corrected data

## 🤝 Contributing

This workflow is designed to be extensible. You can:
- Add new error detection methods
- Customize the web interface
- Integrate with other tools
- Improve the JoyTag model

## 📄 License

This project is open source and available under the MIT License.