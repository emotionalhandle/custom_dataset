#!/usr/bin/env python3
"""
advanced_joytag_processing.py - Extract JoyTag embeddings used by the advanced (unfrozen) training
and cache them to disk per-hash, while mapping hashes → embedding_path and tags.

Differences from process_dataset_with_joytag.py:
- Saves per-image JoyTag embedding to disk (./advanced_embeddings/<hash>.pt) instead of inlining large arrays in JSON
- Adds extra_metadata with person:* and title:* tags

Outputs:
- advanced_joytag_cache.json (optional cache)
- advanced_joytag_complete_<timestamp>.json (mapping + summary)
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import torch
import torchvision.transforms as transforms
from PIL import Image

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Config
try:
    from config import API_URL, API_KEY, REQUEST_TIMEOUT
except ImportError:
    API_URL = "http://127.0.0.1:45869"
    API_KEY = None
    REQUEST_TIMEOUT = 30

HEADERS = {"Hydrus-Client-API-Access-Key": API_KEY} if API_KEY else {}

# Session
SESSION = requests.Session()
if API_KEY:
    SESSION.headers.update(HEADERS)

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=64, max_retries=Retry(total=3, backoff_factor=0.2))
    SESSION.mount("http://", adapter)
    SESSION.mount("https://", adapter)
except Exception:
    pass


def find_most_recent_dataset_file() -> str | None:
    mappings_dir = "mappings"
    if not os.path.exists(mappings_dir):
        print(f"❌ Mappings folder not found: {mappings_dir}")
        return None
    dataset_files: List[Tuple[str, float]] = []  # primary: new naming
    series_files: List[Tuple[str, float]] = []   # series tag mappings
    legacy_all: List[Tuple[str, float]] = []     # legacy: dataset_all_detailed_mapping_
    legacy_hard: List[Tuple[str, float]] = []    # legacy: dataset_hardcore_detailed_mapping_
    for filename in os.listdir(mappings_dir):
        if filename.startswith("dataset_series_mapping_") and filename.endswith(".json"):
            file_path = os.path.join(mappings_dir, filename)
            series_files.append((file_path, os.stat(file_path).st_mtime))
        elif filename.startswith("dataset_mapping_") and filename.endswith(".json"):
            file_path = os.path.join(mappings_dir, filename)
            dataset_files.append((file_path, os.stat(file_path).st_mtime))
        elif filename.startswith("dataset_all_detailed_mapping_") and filename.endswith(".json"):
            file_path = os.path.join(mappings_dir, filename)
            legacy_all.append((file_path, os.stat(file_path).st_mtime))
        elif filename.startswith("dataset_hardcore_detailed_mapping_") and filename.endswith(".json"):
            file_path = os.path.join(mappings_dir, filename)
            legacy_hard.append((file_path, os.stat(file_path).st_mtime))
    
    # Priority order: series mappings > dataset mappings > legacy files
    if series_files:
        series_files.sort(key=lambda x: x[1], reverse=True)
        most_recent = series_files[0][0]
        print(f"📁 Using series tag mapping file: {os.path.basename(most_recent)}")
        return most_recent
    elif dataset_files:
        dataset_files.sort(key=lambda x: x[1], reverse=True)
        most_recent = dataset_files[0][0]
        print(f"📁 Using dataset mapping file: {os.path.basename(most_recent)}")
        return most_recent
    elif legacy_all:
        legacy_all.sort(key=lambda x: x[1], reverse=True)
        most_recent = legacy_all[0][0]
        print("ℹ️  Using legacy 'dataset_all' mapping files (no 'dataset_mapping' or 'series_mapping' files found).")
        print(f"📁 Using file: {os.path.basename(most_recent)}")
        return most_recent
    elif legacy_hard:
        legacy_hard.sort(key=lambda x: x[1], reverse=True)
        most_recent = legacy_hard[0][0]
        print("ℹ️  Using legacy hardcore mapping files (no other mapping files found).")
        print(f"📁 Using file: {os.path.basename(most_recent)}")
        return most_recent
    
    print("❌ No dataset files found in mappings/")
    return None


def load_dataset_file(dataset_path: str) -> tuple[dict | None, List[str]]:
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        all_hashes: List[str] = []
        for _cat, cat_data in dataset.get('categories', {}).items():
            for page in cat_data.get('pages', []):
                all_hashes.extend(page.get('hashes', []))
        return dataset, list(set(all_hashes))
    except Exception as e:
        print(f"❌ Error loading dataset file: {e}")
        return None, []


def fetch_metadata_for_hashes(hashes: List[str]) -> Dict[str, dict]:
    try:
        if not hashes:
            return {}
        resp = SESSION.get(
            f"{API_URL}/get_files/file_metadata",
            params={"hashes": json.dumps(list(hashes))},
            timeout=REQUEST_TIMEOUT
        )
        out: Dict[str, dict] = {}
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get('metadata', []):
                h = item.get('hash')
                if h:
                    out[h] = item
        return out
    except Exception as e:
        print(f"   ❌ Error fetching metadata: {e}")
        return {}


def get_file_path_from_metadata(md: dict) -> str | None:
    try:
        # Try file_path in metadata
        p = md.get('file_path')
        if p and os.path.exists(p):
            return p
        file_id = md.get('file_id')
        if not file_id:
            return None
        # Try file_path by id
        r = SESSION.get(f"{API_URL}/get_files/file_path", params={"file_id": file_id}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            p2 = r.json().get('path', '')
            if p2 and os.path.exists(p2):
                return p2
        # Fallback: fetch file content
        r2 = SESSION.get(f"{API_URL}/get_files/file", params={"file_id": file_id}, timeout=REQUEST_TIMEOUT)
        if r2.status_code == 200:
            temp_path = Path(current_dir) / f"temp_{md.get('hash','unknown')[:8]}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(r2.content)
            return str(temp_path)
    except Exception as e:
        print(f"   ❌ Error resolving file path: {e}")
    return None


def extract_tags_from_md(md: dict) -> tuple[List[str], List[str], List[str]]:
    series_tags: List[str] = []
    all_tags: List[str] = []
    person_tags: List[str] = []
    title_tags: List[str] = []
    try:
        tag_block = (md or {}).get('tags', {})
        for _svc, svc_data in tag_block.items():
            bucket = (svc_data.get('storage_tags') or {}).get("0", [])
            for t in bucket:
                if not isinstance(t, str):
                    continue
                all_tags.append(t)
                tl = t.lower()
                if tl.startswith('series:'):
                    series_tags.append(t)
                if tl.startswith('person:'):
                    person_tags.append(t)
                if tl.startswith('title:'):
                    title_tags.append(t)
        # dedup preserve order
        def uniq(xs: List[str]) -> List[str]:
            seen = set(); out = []
            for x in xs:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        return uniq(series_tags), uniq(all_tags), uniq(person_tags), uniq(title_tags)
    except Exception:
        return series_tags, all_tags, person_tags, title_tags


def _is_descriptive_tag(tag_text: str) -> bool:
    """Heuristic: exclude administrative/identity namespaces, keep content-descriptive tags.

    Examples of excluded namespaces: series:, person:, title:, character:, creator:, artist:,
    url:, source:, file:, filename:, id:, sha256:, md5:, hash:, page:, rating:, date:.
    """
    if not isinstance(tag_text, str) or not tag_text:
        return False
    lt = tag_text.lower()
    # Split once on the first colon to detect namespace
    if ":" in lt:
        ns, _rest = lt.split(":", 1)
        excluded_namespaces = {
            'series', 'person', 'title', 'character', 'creator', 'artist', 'studio', 'publisher',
            'copyright', 'meta', 'system', 'url', 'source', 'file', 'filename', 'id', 'sha256',
            'md5', 'hash', 'page', 'rating', 'date', 'time', 'resolution', 'size', 'format',
            'pixiv', 'danbooru', 'sankaku', 'gelbooru', 'e621', 'derpibooru'
        }
        if ns in excluded_namespaces:
            return False
    # Drop very short tokens that are unlikely to be descriptive
    if len(lt.strip()) <= 1:
        return False
    return True


def compute_tag_metrics(all_tags: List[str]) -> dict:
    """Compute simple tag density metrics from a list of tags.

    Returns a dict with:
      - total_tag_count
      - descriptive_tag_count (excludes administrative namespaces)
      - descriptive_ratio
    """
    try:
        total_tag_count = len(all_tags) if isinstance(all_tags, list) else 0
        descriptive_tag_count = sum(1 for t in (all_tags or []) if _is_descriptive_tag(t))
        descriptive_ratio = (descriptive_tag_count / total_tag_count) if total_tag_count > 0 else 0.0
        return {
            'total_tag_count': int(total_tag_count),
            'descriptive_tag_count': int(descriptive_tag_count),
            'descriptive_ratio': float(descriptive_ratio),
        }
    except Exception:
        return {
            'total_tag_count': 0,
            'descriptive_tag_count': 0,
            'descriptive_ratio': 0.0,
        }


def preprocess_image(path: str) -> torch.Tensor | None:
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        m = max(w, h)
        canvas = Image.new('RGB', (m, m), (255, 255, 255))
        canvas.paste(img, ((m - w)//2, (m - h)//2))
        if m != 448:
            canvas = canvas.resize((448, 448), Image.Resampling.BICUBIC)
        t = transforms.functional.to_tensor(canvas)
        t = transforms.functional.normalize(
            t,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return t
    except Exception as e:
        print(f"   ❌ Preprocess error: {e}")
        return None


def extract_embeddings_batch(paths: List[str], model, device) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    valid_idx: List[int] = []
    for i, p in enumerate(paths):
        if os.path.exists(p):
            t = preprocess_image(p)
            if t is not None:
                tensors.append(t)
                valid_idx.append(i)
    if not tensors:
        return []
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        out = model({"image": batch}, return_embeddings=True)
    emb = out.get('embeddings') if isinstance(out, dict) else None
    if emb is None:
        return []
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu()
    # Return per-item tensors (float32)
    return [emb[i].float() for i in range(emb.shape[0])]


def load_joytag_model(device: torch.device):
    try:
        from joytag.Models import ViT
        model_path = os.path.join(current_dir, 'joytag')
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, 'joytag')
        m = ViT.load_model(model_path)
        return m.to(device).eval()
    except Exception as e:
        print(f"❌ Failed to load JoyTag model: {e}")
        return None


def load_cache(cache_file: Path) -> tuple[List[dict], set]:
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            results = data.get('results', [])
            hashes = {r.get('hash') for r in results}
            return results, hashes
        except Exception as e:
            print(f"⚠️  Could not read cache: {e}")
    return [], set()


def save_cache(cache_file: Path, results: List[dict]):
    try:
        cache_file.write_text(json.dumps({
            'cached_at': datetime.now().isoformat(),
            'total_cached': len(results),
            'results': results
        }, indent=2), encoding='utf-8')
        print(f"💾 Cache updated: {cache_file}")
    except Exception as e:
        print(f"❌ Cache save error: {e}")


def main():
    # Args
    numeric_args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    batch_size = int(numeric_args[0]) if len(numeric_args) > 0 else 20
    max_hashes = int(numeric_args[1]) if len(numeric_args) > 1 else 0
    use_cache = ('--no-cache' not in sys.argv)
    recompute_tags = ('--recompute-tags' in sys.argv)
    force_regenerate = ('--force-regenerate' in sys.argv)
    # Sampling flags (order-preserving random subset)
    sample_count = 0
    sample_frac = 0.0
    sample_seed = None
    for arg in sys.argv[1:]:
        if arg.startswith('--sample='):
            try:
                sample_count = int(arg.split('=', 1)[1])
            except Exception:
                sample_count = 0
        elif arg.startswith('--sample-count='):
            try:
                sample_count = int(arg.split('=', 1)[1])
            except Exception:
                sample_count = 0
        elif arg.startswith('--sample-frac='):
            try:
                sample_frac = float(arg.split('=', 1)[1])
            except Exception:
                sample_frac = 0.0
        elif arg.startswith('--sample-seed='):
            try:
                sample_seed = int(arg.split('=', 1)[1])
            except Exception:
                sample_seed = None

    embeddings_dir = Path('advanced_embeddings')
    embeddings_dir.mkdir(exist_ok=True)
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'advanced_joytag_cache.json'

    # Dataset
    dataset_path = find_most_recent_dataset_file()
    if not dataset_path:
        return
    dataset, all_hashes = load_dataset_file(dataset_path)
    if not all_hashes:
        print("❌ No hashes to process")
        return
    dataset_set = set(all_hashes)
    if max_hashes > 0:
        all_hashes = all_hashes[:max_hashes]
        print(f"🎯 Limiting to first {len(all_hashes)} hashes")

    # Helper: order-preserving random sampling
    def order_preserving_sample(items: list[str]) -> list[str]:
        try:
            n = len(items)
            if n == 0:
                return items
            k_from_frac = int(n * sample_frac) if sample_frac > 0 else 0
            k = max(sample_count, k_from_frac)
            if k <= 0 or k >= n:
                return items
            import random as _random
            rng = _random.Random(sample_seed)
            sel_idx = rng.sample(range(n), k)
            sel_idx.sort()
            return [items[i] for i in sel_idx]
        except Exception:
            return items

    # Cache handling
    existing_results: List[dict] = []
    existing_hashes: set = set()
    if use_cache:
        existing_results, existing_hashes = load_cache(cache_file)
        if existing_hashes:
            # Verify that embedding files actually exist for cached results
            print(f"🔍 Verifying cached embeddings exist...")
            valid_hashes = set()
            for result in existing_results:
                hash_val = result.get('hash')
                if hash_val:
                    emb_path = result.get('embedding_path')
                    if emb_path and Path(emb_path).exists():
                        valid_hashes.add(hash_val)
                    else:
                        # Remove from cache if embedding file doesn't exist
                        print(f"   ⚠️  Missing embedding file for {hash_val[:16]}...")
            
            if len(valid_hashes) != len(existing_hashes):
                print(f"   🧹 Cleaned cache: {len(valid_hashes)}/{len(existing_hashes)} embeddings still exist")
                # Filter results to only include valid ones
                existing_results = [r for r in existing_results if r.get('hash') in valid_hashes]
                existing_hashes = valid_hashes
                # Save cleaned cache
                save_cache(cache_file, existing_results)
            
            # If force regenerate is set, clear the cache and start fresh
            if force_regenerate:
                print(f"   🔄 Force regenerate: clearing cache and starting fresh")
                existing_results = []
                existing_hashes = set()
            
            # Optionally refresh tags for cached entries
            if recompute_tags:
                hashes_list = list(existing_hashes)
                # Restrict recompute to current dataset only
                if dataset_set:
                    hashes_list = [h for h in hashes_list if h in dataset_set]
                # Apply sampling for recompute pass if requested
                if sample_count > 0 or sample_frac > 0:
                    before = len(hashes_list)
                    hashes_list = order_preserving_sample(hashes_list)
                    print(f"🎲 Sampling recompute set: {len(hashes_list)}/{before} (seed={sample_seed if sample_seed is not None else 'auto'})")
                print(f"🔄 Recomputing tags for {len(hashes_list)} cached items (keeping embeddings)")
                idx_map = {r.get('hash'): i for i, r in enumerate(existing_results) if r.get('hash')}
                for i in range(0, len(hashes_list), 500):
                    chunk = hashes_list[i:i+500]
                    md_map = fetch_metadata_for_hashes(chunk)
                    for h, md in md_map.items():
                        try:
                            series, alltags, persons, titles = extract_tags_from_md(md)
                            fp_new = get_file_path_from_metadata(md) or existing_results[idx_map[h]].get('file_path')
                            ix = idx_map.get(h)
                            if ix is None:
                                continue
                            existing_results[ix]['existing_series_tags'] = series
                            existing_results[ix]['existing_all_tags'] = alltags
                            # Update tag metrics on recompute
                            existing_results[ix]['tag_metrics'] = compute_tag_metrics(alltags)
                            existing_results[ix]['extra_metadata'] = {
                                'person_tags': persons,
                                'title_tags': titles
                            }
                            if fp_new:
                                existing_results[ix]['file_path'] = fp_new
                            existing_results[ix]['timestamp'] = datetime.now().isoformat()
                        except Exception:
                            continue
                # Persist refreshed cache
                save_cache(cache_file, existing_results)
            # Skip embedding recompute for cached hashes
            all_hashes = [h for h in all_hashes if h not in existing_hashes]
            print(f"⏭️  Skipping {len(existing_hashes)} already cached hashes. Remaining: {len(all_hashes)}")

    # Apply sampling to remaining new hashes (will preserve original order)
    if sample_count > 0 or sample_frac > 0:
        before = len(all_hashes)
        all_hashes = order_preserving_sample(all_hashes)
        print(f"🎲 Sampling new set: {len(all_hashes)}/{before} (seed={sample_seed if sample_seed is not None else 'auto'})")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    jt = load_joytag_model(device)
    if jt is None:
        return

    # Process in batches
    new_results: List[dict] = []
    for i in range(0, len(all_hashes), batch_size):
        batch_hashes = all_hashes[i:i+batch_size]
        print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(all_hashes) + batch_size - 1)//batch_size}")
        md_map = fetch_metadata_for_hashes(batch_hashes)

        paths: List[str] = []
        per_item: List[dict] = []
        for h in batch_hashes:
            md = md_map.get(h)
            if not md:
                continue
            fp = get_file_path_from_metadata(md)
            if not fp:
                continue
            series, alltags, persons, titles = extract_tags_from_md(md)
            per_item.append({
                'hash': h,
                'file_path': fp,
                'existing_series_tags': series,
                'existing_all_tags': alltags,
                'extra_metadata': {
                    'person_tags': persons,
                    'title_tags': titles
                }
            })
            paths.append(fp)

        if not per_item:
            continue

        # Extract embeddings for this batch
        emb_list = extract_embeddings_batch(paths, jt, device)
        if not emb_list or len(emb_list) != len(per_item):
            print("   ⚠️  Embedding batch mismatch; skipping")
            continue

        # Save embeddings to disk and build results
        for item, emb in zip(per_item, emb_list):
            h = item['hash']
            emb_path = embeddings_dir / f"{h}.pt"
            try:
                torch.save(emb, emb_path)
            except Exception as e:
                print(f"   ❌ Failed saving embedding for {h[:16]}...: {e}")
                continue
            result = {
                'hash': h,
                'file_path': item['file_path'],
                'existing_series_tags': item['existing_series_tags'],
                'existing_all_tags': item['existing_all_tags'],
                'embedding_path': str(emb_path),
                'tag_metrics': compute_tag_metrics(item['existing_all_tags']),
                'extra_metadata': item['extra_metadata'],
                'timestamp': datetime.now().isoformat()
            }
            new_results.append(result)

        # Update cache incrementally
        if use_cache:
            combined = existing_results + new_results
            save_cache(cache_file, combined)

    # Restrict final results to the current dataset mapping to avoid cache poisoning
    def _filter_to_dataset(rows: List[dict]) -> List[dict]:
        try:
            return [r for r in rows if r.get('hash') in dataset_set]
        except Exception:
            return rows
    all_results = _filter_to_dataset(existing_results) + _filter_to_dataset(new_results)
    print(f"\n📊 Combined results (restricted to current dataset): {len(all_results)} total of {len(dataset_set)} dataset hashes")

    # Final JSON
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_file = Path(f"advanced_joytag_complete_{ts}.json")
    # Aggregate tag density summary (optional)
    try:
        ratios = [r.get('tag_metrics', {}).get('descriptive_ratio', 0.0) for r in all_results]
        avg_ratio = float(sum(ratios) / max(1, len(ratios))) if ratios else 0.0
    except Exception:
        avg_ratio = 0.0
    summary = {
        'total_hashes_processed': len(all_results),
        'hashes_with_series_tags': len([r for r in all_results if r.get('existing_series_tags')]),
        'hashes_with_embeddings': len([r for r in all_results if r.get('embedding_path')]),
        'avg_descriptive_tag_ratio': avg_ratio,
        'timestamp': datetime.now().isoformat(),
        'embeddings_dir': str(embeddings_dir)
    }
    final_file.write_text(json.dumps({'summary': summary, 'results': all_results}, indent=2), encoding='utf-8')
    print(f"📁 Final results saved to: {final_file}")


if __name__ == '__main__':
    main()