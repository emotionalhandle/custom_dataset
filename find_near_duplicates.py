#!/usr/bin/env python3
"""
find_near_duplicates.py - Find near-duplicate images using JoyTag embeddings
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import time

def load_embeddings(embeddings_dir: str, max_embeddings: int = 1000) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from the advanced_embeddings directory."""
    embeddings_dir = Path(embeddings_dir)
    embedding_files = list(embeddings_dir.glob("*.pt"))
    
    if max_embeddings > 0:
        embedding_files = embedding_files[:max_embeddings]
    
    hashes = []
    embeddings = []
    
    print(f"📁 Loading {len(embedding_files)} embeddings...")
    
    for i, embedding_file in enumerate(embedding_files):
        if i % 100 == 0:
            print(f"   Loading {i}/{len(embedding_files)}")
        
        try:
            embedding = torch.load(embedding_file, map_location='cpu')
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.numpy()
            elif isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding)
            
            # Ensure embedding is 1D
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            hashes.append(embedding_file.stem)
            embeddings.append(embedding)
            
        except Exception as e:
            print(f"   ⚠️  Error loading {embedding_file}: {e}")
            continue
    
    embeddings_array = np.array(embeddings)
    print(f"✅ Loaded {len(hashes)} embeddings with shape {embeddings_array.shape}")
    
    return hashes, embeddings_array

def find_similar_pairs(hashes: List[str], embeddings: np.ndarray, 
                      similarity_threshold: float = 0.95) -> List[Dict]:
    """Find pairs of similar images using cosine similarity."""
    print(f"🔍 Computing similarity matrix for {len(hashes)} images...")
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find pairs above threshold (excluding self-similarity)
    similar_pairs = []
    
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                similar_pairs.append({
                    'hash1': hashes[i],
                    'hash2': hashes[j],
                    'similarity': float(similarity),
                    'index1': i,
                    'index2': j
                })
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"✅ Found {len(similar_pairs)} similar pairs above threshold {similarity_threshold}")
    
    return similar_pairs

def load_tag_data(dataset_file: str) -> Dict[str, Dict]:
    """Load tag data from the processed dataset."""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    results = dataset.get('results', [])
    tag_data = {}
    
    for result in results:
        hash_val = result.get('hash')
        if hash_val:
            tag_data[hash_val] = {
                'series_tags': result.get('existing_series_tags', []),
                'all_tags': result.get('existing_all_tags', []),
                'person_tags': result.get('extra_metadata', {}).get('person_tags', []),
                'file_path': result.get('file_path', '')
            }
    
    return tag_data

def analyze_tag_differences(pair: Dict, tag_data: Dict[str, Dict]) -> Dict:
    """Analyze tag differences between similar images."""
    hash1, hash2 = pair['hash1'], pair['hash2']
    
    data1 = tag_data.get(hash1, {})
    data2 = tag_data.get(hash2, {})
    
    if not data1 or not data2:
        return {'error': 'Missing tag data'}
    
    # Compare series tags
    series1 = set(data1.get('series_tags', []))
    series2 = set(data2.get('series_tags', []))
    
    # Compare person tags
    person1 = set(data1.get('person_tags', []))
    person2 = set(data2.get('person_tags', []))
    
    # Compare all tags
    all1 = set(data1.get('all_tags', []))
    all2 = set(data2.get('all_tags', []))
    
    return {
        'similarity': pair['similarity'],
        'series_tags_common': list(series1 & series2),
        'series_tags_diff': {
            'only_in_1': list(series1 - series2),
            'only_in_2': list(series2 - series1)
        },
        'person_tags_common': list(person1 & person2),
        'person_tags_diff': {
            'only_in_1': list(person1 - person2),
            'only_in_2': list(person2 - person1)
        },
        'all_tags_common': list(all1 & all2),
        'all_tags_diff': {
            'only_in_1': list(all1 - all2),
            'only_in_2': list(all2 - all1)
        },
        'file_paths': {
            'hash1': data1.get('file_path', ''),
            'hash2': data2.get('file_path', '')
        }
    }

def generate_similarity_report(similar_pairs: List[Dict], tag_data: Dict[str, Dict], 
                              output_file: str, max_pairs: int = 100):
    """Generate a report of similar images with tag analysis."""
    print(f"📊 Analyzing tag differences for {min(len(similar_pairs), max_pairs)} pairs...")
    
    report = {
        'summary': {
            'total_pairs_found': len(similar_pairs),
            'pairs_analyzed': min(len(similar_pairs), max_pairs),
            'similarity_threshold': 0.95
        },
        'similar_pairs': []
    }
    
    for i, pair in enumerate(similar_pairs[:max_pairs]):
        if i % 10 == 0:
            print(f"   Analyzing pair {i}/{min(len(similar_pairs), max_pairs)}")
        
        analysis = analyze_tag_differences(pair, tag_data)
        if 'error' not in analysis:
            report['similar_pairs'].append({
                'hash1': pair['hash1'],
                'hash2': pair['hash2'],
                'similarity': pair['similarity'],
                'analysis': analysis
            })
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    # Configuration
    embeddings_dir = "advanced_embeddings"
    similarity_threshold = 0.95
    max_embeddings = 2000  # Limit for performance
    
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
    
    # Load embeddings
    if not os.path.exists(embeddings_dir):
        print(f"❌ Embeddings directory not found: {embeddings_dir}")
        return
    
    hashes, embeddings = load_embeddings(embeddings_dir, max_embeddings)
    
    if len(hashes) == 0:
        print("❌ No embeddings found")
        return
    
    # Find similar pairs
    similar_pairs = find_similar_pairs(hashes, embeddings, similarity_threshold)
    
    if not similar_pairs:
        print("✅ No similar pairs found above threshold")
        return
    
    # Load tag data
    print("📁 Loading tag data...")
    tag_data = load_tag_data(dataset_file)
    
    # Generate report
    output_file = f"near_duplicates_report_{int(time.time())}.json"
    report = generate_similarity_report(similar_pairs, tag_data, output_file)
    
    # Print summary
    print("\n📊 SIMILARITY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total similar pairs found: {len(similar_pairs)}")
    print(f"Pairs analyzed: {len(report['similar_pairs'])}")
    
    # Show top similar pairs
    print("\n🔍 TOP SIMILAR PAIRS:")
    for i, pair in enumerate(report['similar_pairs'][:10]):
        print(f"  {i+1}. {pair['hash1'][:16]}... <-> {pair['hash2'][:16]}... (similarity: {pair['similarity']:.3f})")
        
        analysis = pair['analysis']
        if analysis['series_tags_diff']['only_in_1'] or analysis['series_tags_diff']['only_in_2']:
            print(f"     Series tag differences:")
            if analysis['series_tags_diff']['only_in_1']:
                print(f"       Only in {pair['hash1'][:16]}: {analysis['series_tags_diff']['only_in_1']}")
            if analysis['series_tags_diff']['only_in_2']:
                print(f"       Only in {pair['hash2'][:16]}: {analysis['series_tags_diff']['only_in_2']}")
    
    print(f"\n📁 Detailed report saved to: {output_file}")

if __name__ == '__main__':
    main()