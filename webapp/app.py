#!/usr/bin/env python3
"""
Flask web application for displaying image predictions
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, g, send_file, abort
import threading
import subprocess
import time
import json
import os
from pathlib import Path
import base64
import sys
import sqlite3
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import colorsys
import shutil
try:
    import requests
except Exception:
    requests = None

# Ensure parent directory is on path to import config and hydrus helpers
PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from config import API_URL, API_KEY, REQUEST_TIMEOUT
except Exception:
    API_URL = os.environ.get('HYDRUS_API_URL', 'http://127.0.0.1:45869')
    API_KEY = os.environ.get('HYDRUS_API_KEY')
    REQUEST_TIMEOUT = int(os.environ.get('HYDRUS_REQUEST_TIMEOUT', '30'))

try:
    from config import MODEL_API_URL
except Exception:
    MODEL_API_URL = os.environ.get('MODEL_API_URL', '')

try:
    from hydrus_tag_management.hydrus_tag_manager import HydrusTagManager, HydrusConfig
    _HYDRUS_AVAILABLE = True
except Exception:
    _HYDRUS_AVAILABLE = False

# Import the image path function from advanced_predict
try:
    from advanced_predict import get_image_from_hydrus
except Exception:
    def get_image_from_hydrus(image_hash):
        return None

# CLIP Series Classifier model loading
_CLIP_CLASSIFIER_MODEL = None
_CLIP_CLASSIFIER_DEVICE = None
_CLIP_CLASSIFIER_TAG_NAMES = None
_CLIP_PROCESSOR = None
_CLIP_VISION_MODEL = None


def load_clip_classifier_model():
    """Load CLIP-based series classifier model."""
    global _CLIP_CLASSIFIER_MODEL, _CLIP_CLASSIFIER_DEVICE, _CLIP_CLASSIFIER_TAG_NAMES, _CLIP_PROCESSOR, _CLIP_VISION_MODEL
    if _CLIP_CLASSIFIER_MODEL is not None:
        return _CLIP_CLASSIFIER_MODEL, _CLIP_CLASSIFIER_DEVICE, _CLIP_CLASSIFIER_TAG_NAMES, _CLIP_PROCESSOR, _CLIP_VISION_MODEL
    
    print("⚠️  CLIP classifier disabled - using JoyTag embeddings instead")
    return None, None, None, None, None

def predict_clip_series_tags(image_path, limit=20):
    """Predict series tags using CLIP-based classifier - DISABLED (using JoyTag instead)."""
    print("⚠️  CLIP series predictions disabled - using JoyTag instead")
    return []

app = Flask(__name__)

# Configuration
IMAGES_PER_PAGE = 48  # 8 columns × 6 rows
THUMBNAIL_SIZE = 150
MAX_IMAGES = 1000     # maximum non-archived images loaded at once on initial page

# Directories
TEMP_IMAGES_DIR = Path('temp_images')
TEMP_THUMBS_DIR = Path('temp_thumbnails')
TEMP_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR = Path('annotations')
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Background pipeline state
_pipeline_lock = threading.Lock()
_pipeline_running = False

def _run_pipeline_background(use_quick: bool = False):
    global _pipeline_running
    try:
        cmd = [sys.executable, 'run_all.py']
        if use_quick:
            cmd.append('--quick')
        subprocess.run(cmd, cwd=PARENT_DIR, check=True)
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        with _pipeline_lock:
            _pipeline_running = False

@app.route('/')
def index():
    """Main page showing image predictions."""
    return render_template('index.html')

@app.route('/api/images')
def api_images():
    """API endpoint to get images with predictions."""
    try:
        # Get database path
        db_path = Path('predictions.db')
        if not db_path.exists():
            return jsonify({'error': 'Database not found'}), 404
        
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', IMAGES_PER_PAGE))
        tag_filter = request.args.get('tag', '')
        sort_by = request.args.get('sort', 'confidence')
        
        # Build query
        offset = (page - 1) * limit
        where_clause = ""
        params = []
        
        if tag_filter:
            where_clause = "WHERE tags LIKE ?"
            params.append(f"%{tag_filter}%")
        
        # Order by clause
        order_clause = "ORDER BY confidence DESC"
        if sort_by == 'hash':
            order_clause = "ORDER BY hash"
        elif sort_by == 'tags':
            order_clause = "ORDER BY tags"
        
        query = f"""
        SELECT hash, tags, confidence, file_path
        FROM predictions 
        {where_clause}
        {order_clause}
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        images = []
        for row in rows:
            images.append({
                'hash': row['hash'],
                'tags': row['tags'],
                'confidence': row['confidence'],
                'file_path': row['file_path']
            })
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM predictions {where_clause}"
        cursor.execute(count_query, params[:-2])  # Remove limit and offset
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'images': images,
            'total': total,
            'page': page,
            'limit': limit
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_pipeline', methods=['POST'])
def run_pipeline():
    """Start the prediction pipeline in the background."""
    global _pipeline_running
    
    with _pipeline_lock:
        if _pipeline_running:
            return jsonify({'status': 'already_running'})
        
        _pipeline_running = True
    
    # Start background thread
    use_quick = request.json.get('quick', False) if request.is_json else False
    thread = threading.Thread(target=_run_pipeline_background, args=(use_quick,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/pipeline_status')
def pipeline_status():
    """Check if pipeline is running."""
    with _pipeline_lock:
        return jsonify({'running': _pipeline_running})

@app.route('/image/<hash>')
def get_image(hash):
    """Serve image by hash."""
    try:
        # Try to get image from Hydrus
        image_path = get_image_from_hydrus(hash)
        if image_path and os.path.exists(image_path):
            return send_file(image_path)
        
        # Fallback to temp images
        temp_path = TEMP_IMAGES_DIR / f"{hash}.jpg"
        if temp_path.exists():
            return send_file(temp_path)
        
        return abort(404)
    except Exception as e:
        return abort(500)

@app.route('/thumbnail/<hash>')
def get_thumbnail(hash):
    """Serve thumbnail by hash."""
    try:
        thumb_path = TEMP_THUMBS_DIR / f"{hash}.jpg"
        if thumb_path.exists():
            return send_file(thumb_path)
        
        # Generate thumbnail if it doesn't exist
        image_path = get_image_from_hydrus(hash)
        if image_path and os.path.exists(image_path):
            from PIL import Image
            img = Image.open(image_path)
            img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
            img.save(thumb_path, 'JPEG', quality=85)
            return send_file(thumb_path)
        
        return abort(404)
    except Exception as e:
        return abort(500)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)