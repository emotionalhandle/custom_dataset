#!/usr/bin/env python3
"""
validate_series_predictions.py - Validate series classifier predictions against existing tags
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_dataset(dataset_file: str) -> Dict:
    """Load the processed dataset with predictions."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def extract_series_tags_from_metadata(metadata: Dict) -> List[str]:
    """Extract series tags from metadata."""
    series_tags = []
    
    # Check existing_series_tags
    if 'existing_series_tags' in metadata:
        series_tags.extend(metadata['existing_series_tags'])
    
    # Check all_tags for series patterns
    if 'existing_all_tags' in metadata:
        all_tags = metadata['existing_all_tags']
        # Look for common series patterns
        series_patterns = ['series:', 'studio:', 'brand:', 'franchise:']
        for tag in all_tags:
            if any(pattern in tag.lower() for pattern in series_patterns):
                series_tags.append(tag)
    
    return list(set(series_tags))  # Remove duplicates

def analyze_prediction_accuracy(dataset: Dict, confidence_threshold: float = 0.5) -> Dict:
    """Analyze the accuracy of series predictions."""
    results = dataset.get('results', [])
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    confidence_scores = []
    hash_list = []
    
    print(f"📊 Analyzing {len(results)} predictions...")
    
    for i, result in enumerate(results):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(results)}")
        
        hash_val = result.get('hash')
        if not hash_val:
            continue
        
        # Get predicted series
        predicted_series = result.get('predicted_series', [])
        if not predicted_series:
            continue
        
        # Get ground truth series tags
        existing_series = extract_series_tags_from_metadata(result)
        
        # For each predicted series, check if it matches ground truth
        for pred_series in predicted_series:
            series_name = pred_series.get('series', '')
            confidence = pred_series.get('confidence', 0.0)
            
            if confidence >= confidence_threshold:
                # Check if this series exists in ground truth
                is_correct = series_name in existing_series
                
                predictions.append(1 if is_correct else 0)
                ground_truth.append(1)  # We're predicting this series exists
                confidence_scores.append(confidence)
                hash_list.append(hash_val)
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'confidence_scores': confidence_scores,
        'hash_list': hash_list,
        'total_predictions': len(predictions),
        'correct_predictions': sum(predictions),
        'accuracy': sum(predictions) / len(predictions) if predictions else 0
    }

def find_missing_series_predictions(dataset: Dict, confidence_threshold: float = 0.5) -> List[Dict]:
    """Find series that exist in tags but weren't predicted."""
    results = dataset.get('results', [])
    missing_predictions = []
    
    print(f"🔍 Looking for missing series predictions...")
    
    for i, result in enumerate(results):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(results)}")
        
        hash_val = result.get('hash')
        if not hash_val:
            continue
        
        # Get existing series tags
        existing_series = extract_series_tags_from_metadata(result)
        if not existing_series:
            continue
        
        # Get predicted series
        predicted_series = result.get('predicted_series', [])
        predicted_series_names = [pred.get('series', '') for pred in predicted_series 
                                if pred.get('confidence', 0) >= confidence_threshold]
        
        # Find missing predictions
        for series in existing_series:
            if series not in predicted_series_names:
                missing_predictions.append({
                    'hash': hash_val,
                    'missing_series': series,
                    'existing_series': existing_series,
                    'predicted_series': predicted_series_names,
                    'file_path': result.get('file_path', '')
                })
    
    return missing_predictions

def find_incorrect_series_predictions(dataset: Dict, confidence_threshold: float = 0.5) -> List[Dict]:
    """Find series that were predicted but don't exist in tags."""
    results = dataset.get('results', [])
    incorrect_predictions = []
    
    print(f"🔍 Looking for incorrect series predictions...")
    
    for i, result in enumerate(results):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(results)}")
        
        hash_val = result.get('hash')
        if not hash_val:
            continue
        
        # Get existing series tags
        existing_series = extract_series_tags_from_metadata(result)
        
        # Get predicted series
        predicted_series = result.get('predicted_series', [])
        
        # Find incorrect predictions
        for pred_series in predicted_series:
            series_name = pred_series.get('series', '')
            confidence = pred_series.get('confidence', 0.0)
            
            if confidence >= confidence_threshold and series_name not in existing_series:
                incorrect_predictions.append({
                    'hash': hash_val,
                    'incorrect_series': series_name,
                    'confidence': confidence,
                    'existing_series': existing_series,
                    'file_path': result.get('file_path', '')
                })
    
    return incorrect_predictions

def generate_validation_report(dataset: Dict, output_file: str, confidence_threshold: float = 0.5):
    """Generate a comprehensive validation report."""
    print("📊 Generating validation report...")
    
    # Analyze prediction accuracy
    accuracy_analysis = analyze_prediction_accuracy(dataset, confidence_threshold)
    
    # Find missing predictions
    missing_predictions = find_missing_series_predictions(dataset, confidence_threshold)
    
    # Find incorrect predictions
    incorrect_predictions = find_incorrect_series_predictions(dataset, confidence_threshold)
    
    # Create report
    report = {
        'summary': {
            'total_images': len(dataset.get('results', [])),
            'confidence_threshold': confidence_threshold,
            'total_predictions': accuracy_analysis['total_predictions'],
            'correct_predictions': accuracy_analysis['correct_predictions'],
            'accuracy': accuracy_analysis['accuracy'],
            'missing_predictions_count': len(missing_predictions),
            'incorrect_predictions_count': len(incorrect_predictions)
        },
        'accuracy_analysis': accuracy_analysis,
        'missing_predictions': missing_predictions[:100],  # Limit for file size
        'incorrect_predictions': incorrect_predictions[:100],  # Limit for file size
        'confidence_distribution': {
            'high_confidence': len([c for c in accuracy_analysis['confidence_scores'] if c >= 0.8]),
            'medium_confidence': len([c for c in accuracy_analysis['confidence_scores'] if 0.5 <= c < 0.8]),
            'low_confidence': len([c for c in accuracy_analysis['confidence_scores'] if c < 0.5])
        }
    }
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    # Find the most recent processed dataset
    dataset_files = []
    for file in os.listdir('.'):
        if file.startswith('advanced_joytag_complete_') and file.endswith('.json'):
            dataset_files.append(file)
    
    if not dataset_files:
        print("❌ No processed dataset files found")
        return
    
    dataset_file = sorted(dataset_files)[-1]
    print(f"📁 Using dataset: {dataset_file}")
    
    # Load dataset
    dataset = load_processed_dataset(dataset_file)
    
    # Generate validation report
    output_file = f"series_validation_report_{int(time.time())}.json"
    report = generate_validation_report(dataset, output_file, confidence_threshold=0.5)
    
    # Print summary
    print("\n📊 SERIES PREDICTION VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total images: {report['summary']['total_images']}")
    print(f"Total predictions: {report['summary']['total_predictions']}")
    print(f"Correct predictions: {report['summary']['correct_predictions']}")
    print(f"Accuracy: {report['summary']['accuracy']:.3f}")
    print(f"Missing predictions: {report['summary']['missing_predictions_count']}")
    print(f"Incorrect predictions: {report['summary']['incorrect_predictions_count']}")
    
    # Show confidence distribution
    conf_dist = report['confidence_distribution']
    print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
    print(f"  High confidence (≥0.8): {conf_dist['high_confidence']}")
    print(f"  Medium confidence (0.5-0.8): {conf_dist['medium_confidence']}")
    print(f"  Low confidence (<0.5): {conf_dist['low_confidence']}")
    
    # Show top missing predictions
    if report['missing_predictions']:
        print(f"\n❌ TOP MISSING PREDICTIONS:")
        for i, missing in enumerate(report['missing_predictions'][:10]):
            print(f"  {i+1}. {missing['hash'][:16]}... - Missing: {missing['missing_series']}")
    
    # Show top incorrect predictions
    if report['incorrect_predictions']:
        print(f"\n⚠️  TOP INCORRECT PREDICTIONS:")
        for i, incorrect in enumerate(report['incorrect_predictions'][:10]):
            print(f"  {i+1}. {incorrect['hash'][:16]}... - Incorrect: {incorrect['incorrect_series']} (conf: {incorrect['confidence']:.3f})")
    
    print(f"\n📁 Detailed report saved to: {output_file}")

if __name__ == '__main__':
    import time
    main()