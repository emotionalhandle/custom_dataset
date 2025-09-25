#!/usr/bin/env python3
"""
find_mistagged_images.py - Detect potential tagging errors in the dataset
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import re

def load_processed_dataset(dataset_file: str) -> Dict:
    """Load the processed JoyTag dataset."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_tag_consistency(results: List[Dict]) -> Dict:
    """Analyze tag consistency across the dataset."""
    tag_frequency = Counter()
    series_tag_frequency = Counter()
    person_tag_frequency = Counter()
    
    # Collect tag statistics
    for result in results:
        all_tags = result.get('existing_all_tags', [])
        series_tags = result.get('existing_series_tags', [])
        person_tags = result.get('extra_metadata', {}).get('person_tags', [])
        
        for tag in all_tags:
            tag_frequency[tag] += 1
        for tag in series_tags:
            series_tag_frequency[tag] += 1
        for tag in person_tags:
            person_tag_frequency[tag] += 1
    
    return {
        'tag_frequency': dict(tag_frequency.most_common(50)),
        'series_tag_frequency': dict(series_tag_frequency.most_common(50)),
        'person_tag_frequency': dict(person_tag_frequency.most_common(50))
    }

def find_potential_errors(results: List[Dict]) -> Dict:
    """Find potential tagging errors."""
    errors = {
        'low_tag_count': [],
        'high_tag_count': [],
        'inconsistent_series': [],
        'missing_person_tags': [],
        'suspicious_combinations': [],
        'duplicate_hashes': []
    }
    
    # Track hashes to find duplicates
    hash_counts = Counter()
    
    for result in results:
        hash_val = result.get('hash')
        all_tags = result.get('existing_all_tags', [])
        series_tags = result.get('existing_series_tags', [])
        person_tags = result.get('extra_metadata', {}).get('person_tags', [])
        tag_metrics = result.get('tag_metrics', {})
        
        # Count hash occurrences
        hash_counts[hash_val] += 1
        
        # Low tag count (potential under-tagging)
        total_tags = len(all_tags)
        if total_tags < 3:
            errors['low_tag_count'].append({
                'hash': hash_val,
                'tag_count': total_tags,
                'tags': all_tags
            })
        
        # High tag count (potential over-tagging)
        if total_tags > 50:
            errors['high_tag_count'].append({
                'hash': hash_val,
                'tag_count': total_tags,
                'tags': all_tags[:20]  # Show first 20 tags
            })
        
        # Missing person tags for series content
        if series_tags and not person_tags:
            errors['missing_person_tags'].append({
                'hash': hash_val,
                'series_tags': series_tags,
                'all_tags': all_tags
            })
        
        # Suspicious tag combinations
        suspicious_patterns = [
            (r'series:.*anal', r'series:.*vaginal'),  # Both anal and vaginal
            (r'series:.*solo', r'series:.*group'),     # Solo and group
            (r'series:.*hardcore', r'series:.*softcore'),  # Hardcore and softcore
        ]
        
        for pattern1, pattern2 in suspicious_patterns:
            has_pattern1 = any(re.search(pattern1, tag, re.IGNORECASE) for tag in series_tags)
            has_pattern2 = any(re.search(pattern2, tag, re.IGNORECASE) for tag in series_tags)
            
            if has_pattern1 and has_pattern2:
                errors['suspicious_combinations'].append({
                    'hash': hash_val,
                    'pattern1': pattern1,
                    'pattern2': pattern2,
                    'series_tags': series_tags
                })
    
    # Find duplicate hashes
    for hash_val, count in hash_counts.items():
        if count > 1:
            errors['duplicate_hashes'].append({
                'hash': hash_val,
                'count': count
            })
    
    return errors

def find_similar_images(results: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """Find potentially similar images that might be duplicates."""
    # This would require loading embeddings and computing similarity
    # For now, return empty list as placeholder
    return []

def generate_error_report(errors: Dict, stats: Dict, output_file: str):
    """Generate a comprehensive error report."""
    report = {
        'summary': {
            'total_errors_found': sum(len(error_list) for error_list in errors.values()),
            'low_tag_count': len(errors['low_tag_count']),
            'high_tag_count': len(errors['high_tag_count']),
            'inconsistent_series': len(errors['inconsistent_series']),
            'missing_person_tags': len(errors['missing_person_tags']),
            'suspicious_combinations': len(errors['suspicious_combinations']),
            'duplicate_hashes': len(errors['duplicate_hashes'])
        },
        'tag_statistics': stats,
        'errors': errors
    }
    
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
    
    # Use the most recent file
    dataset_file = sorted(dataset_files)[-1]
    print(f"📁 Using dataset: {dataset_file}")
    
    # Load dataset
    dataset = load_processed_dataset(dataset_file)
    results = dataset.get('results', [])
    print(f"📊 Analyzing {len(results)} images")
    
    # Analyze tag consistency
    print("🔍 Analyzing tag consistency...")
    stats = analyze_tag_consistency(results)
    
    # Find potential errors
    print("🔍 Finding potential tagging errors...")
    errors = find_potential_errors(results)
    
    # Generate report
    output_file = f"tagging_errors_report_{int(time.time())}.json"
    report = generate_error_report(errors, stats, output_file)
    
    # Print summary
    print("\n📊 ERROR ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total errors found: {report['summary']['total_errors_found']}")
    print(f"Low tag count (< 3 tags): {report['summary']['low_tag_count']}")
    print(f"High tag count (> 50 tags): {report['summary']['high_tag_count']}")
    print(f"Missing person tags: {report['summary']['missing_person_tags']}")
    print(f"Suspicious combinations: {report['summary']['suspicious_combinations']}")
    print(f"Duplicate hashes: {report['summary']['duplicate_hashes']}")
    
    print(f"\n📁 Detailed report saved to: {output_file}")
    
    # Show top problematic tags
    print("\n🏷️  TOP SERIES TAGS:")
    for tag, count in list(stats['series_tag_frequency'].items())[:10]:
        print(f"  {tag}: {count}")
    
    print("\n👥 TOP PERSON TAGS:")
    for tag, count in list(stats['person_tag_frequency'].items())[:10]:
        print(f"  {tag}: {count}")

if __name__ == '__main__':
    import time
    main()