#!/usr/bin/env python3
"""
dataset_hardcore_mapper.py - Map all pages under Dataset to hashes using corrected logic.

This script targets the top-level 'Dataset' page and recursively explores ALL sub-pages,
extracting hashes from each individual media page found (no longer limited to 'hardcore').
"""

import requests
import json
import time
import os
import argparse

# Import configuration
try:
    from config import API_URL, API_KEY, REQUEST_TIMEOUT
except ImportError:
    API_URL = "http://127.0.0.1:45869"
    API_KEY = "d0e47ec776ddba336b869e40c4f6d35514eefc7d6d33f25e2c7f2c7b7ab2"
    REQUEST_TIMEOUT = 30

HEADERS = {"Hydrus-Client-API-Access-Key": API_KEY}

def get_page_structure():
    """Get the page structure from Hydrus."""
    try:
        print("Fetching page structure from /manage_pages/get_pages...")
        resp = requests.get(f"{API_URL}/manage_pages/get_pages", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        
        if resp.status_code == 200:
            print("✓ Successfully retrieved page structure!")
            return resp.json()
        else:
            print(f"✗ Failed to get page structure: {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"Error getting page structure: {e}")
        return None

def refresh_page(page_key):
    """Refresh a page in Hydrus (like pressing F5)."""
    try:
        resp = requests.post(
            f"{API_URL}/manage_pages/refresh_page",
            headers={**HEADERS, 'Content-Type': 'application/json'},
            json={"page_key": page_key},
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 404:
            print(f"✗ Refresh 404 for page {page_key[:16]}...")
            return False
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Error refreshing page {page_key[:16]}...: {e}")
        return False

def wait_for_page_complete(page_key, poll_interval=1.0, max_wait=300):
    """Poll page info until the page reports not searching or timeout."""
    start = time.time()
    while True:
        try:
            info = get_page_info(page_key)
            if not info:
                break
            state = {}
            if isinstance(info, dict):
                # Direct state
                if isinstance(info.get('page_state'), dict):
                    state = info.get('page_state') or {}
                # Nested under page_info
                elif isinstance(info.get('page_info'), dict):
                    pi = info.get('page_info') or {}
                    if isinstance(pi.get('page_state'), dict):
                        state = pi.get('page_state') or {}
            searching = bool(state.get('searching', False))
            status_text = state.get('status_text', '')
            print(f"      state: searching={searching} {status_text}")
            if not searching:
                return True
        except Exception as e:
            print(f"      warn: polling error: {e}")
        if time.time() - start > max_wait:
            print("      warn: wait timeout")
            return False
        time.sleep(poll_interval)

def find_dataset_hardcore_structure(page_data, page_hierarchy=None):
    """Find ALL media pages under the specified page hierarchy (defaults to 'dataset')."""
    if page_hierarchy is None:
        page_hierarchy = ['dataset']
    
    def search_recursive(pages, path="", target_hierarchy=None, current_depth=0):
        results = []

        if not pages:
            return results

        for page in pages:
            current_path = f"{path}/{page.get('name', 'Unknown')}" if path else page.get('name', 'Unknown')
            page_name = page.get('name', '')
            page_name_lower = page_name.lower()

            # Check if we're at the right level in the hierarchy
            if target_hierarchy and current_depth < len(target_hierarchy):
                target_page = target_hierarchy[current_depth].lower()
                if page_name_lower == target_page:
                    print(f"🎯 Found {target_page} at: {current_path}")
                    # If this is the last page in hierarchy, start collecting from here
                    if current_depth == len(target_hierarchy) - 1:
                        # Recurse into this page's children to collect media pages
                        if page.get('page_type') == 10 and 'pages' in page and page['pages']:
                            results.extend(search_recursive(page['pages'], current_path, None, current_depth + 1))
                    else:
                        # Continue down the hierarchy
                        if page.get('page_type') == 10 and 'pages' in page and page['pages']:
                            results.extend(search_recursive(page['pages'], current_path, target_hierarchy, current_depth + 1))
            elif not target_hierarchy:
                # No specific hierarchy - collect all media pages
                if page.get('is_media_page', False):
                    # Determine a category as the first segment after the target in the path
                    parts = current_path.split('/')
                    parts_lower = [p.lower() for p in parts]
                    category_name = "root"
                    if len(parts) > 1:
                        category_name = parts[1]

                    print(f"   📄 Found media page: {current_path} (category: {category_name})")
                    results.append((page, current_path, category_name))
                
                # Recurse into containers (Page of pages)
                elif page.get('page_type') == 10 and 'pages' in page and page['pages']:
                    results.extend(search_recursive(page['pages'], current_path, None, current_depth + 1))

        return results

    # Start search from the top level
    if 'pages' in page_data:
        top_pages = page_data['pages']
        if isinstance(top_pages, list):
            return search_recursive(top_pages, "", page_hierarchy, 0)
        elif isinstance(top_pages, dict) and 'pages' in top_pages:
            return search_recursive(top_pages['pages'], "", page_hierarchy, 0)

    return []

def get_page_info(page_key):
    """Get detailed information about a specific page."""
    try:
        # Use simple=false to get actual hashes directly
        resp = requests.get(
            f"{API_URL}/manage_pages/get_page_info",
            headers=HEADERS,
            params={"page_key": page_key, "simple": "false"},
            timeout=REQUEST_TIMEOUT
        )
        
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"✗ Failed to get page info for {page_key[:16]}...: {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"Error getting page info for {page_key[:16]}...: {e}")
        return None

def extract_hashes_from_page(page_key, page_name):
    """Extract hashes from a specific page using corrected logic."""
    # Ensure page is up-to-date before extracting
    print(f"      ↻ Refreshing page '{page_name}'...")
    ok = refresh_page(page_key)
    if ok:
        # Small delay after refresh to allow Hydrus to begin searching
        time.sleep(0.1)
        wait_for_page_complete(page_key)
    else:
        print("      ⚠️  Refresh failed; proceeding with current contents")

    page_info = get_page_info(page_key)
    if not page_info:
        return None
    
    # CORRECTED: Extract media information from inside page_info
    page_info_data = page_info.get('page_info', {})
    media_info = page_info_data.get('media', {})
    hash_count = media_info.get('num_files', 0)
    
    if hash_count == 0:
        return None
    
    # Get hashes directly from detailed response
    hashes = media_info.get('hashes', [])
    
    # Fallback: if no hashes, try to convert hash_ids
    if not hashes:
        hash_ids = media_info.get('hash_ids', [])
        if hash_ids:
            try:
                resp = requests.get(
                    f"{API_URL}/get_files/file_metadata",
                    headers=HEADERS,
                    params={"file_ids": json.dumps(hash_ids)},  # Use file_ids instead of hash_ids
                    timeout=REQUEST_TIMEOUT
                )
                
                if resp.status_code == 200:
                    metadata = resp.json()
                    if 'metadata' in metadata:
                        for item in metadata['metadata']:
                            if 'hash' in item:
                                hashes.append(item['hash'])
            except Exception as e:
                print(f"Error converting hash IDs for {page_name}: {e}")
    
    return {
        "page_name": page_name,
        "page_key": page_key,
        "hash_count": hash_count,
        "hashes": hashes
    }

def create_mapping_file(category_results):
    """Create a mapping file for all media pages under the 'Dataset' page."""
    print("\n📊 CREATING MAPPING FILE")
    print("=" * 50)

    mapping_data = {
        "metadata": {
            "timestamp": int(time.time()),
            "description": "Dataset structure mapping - ALL individual media pages under 'Dataset'",
            "api_url": API_URL
        },
        "categories": {},
        "individual_pages": []
    }

    total_found = 0
    aggregated_hashes = []  # collect all hashes across pages for duplicate analysis
    hash_to_pages = {}  # map hash -> list of page paths where it appears
    
    # Group results by category
    category_groups = {}
    for page, path, category_name in category_results:
        if category_name not in category_groups:
            category_groups[category_name] = []
        category_groups[category_name].append((page, path))
    
    # Process each category
    for category_name, pages in category_groups.items():
        print(f"\n🔍 Processing {category_name} category...")
        print(f"   Found {len(pages)} individual pages under this category")
        
        category_total = 0
        category_pages = []
        
        for page, path in pages:
            print(f"   📄 {path}")
            print(f"      Page key: {page.get('page_key', 'Unknown')[:16]}...")
            
            # Extract hashes from this individual page
            hash_data = extract_hashes_from_page(
                page.get('page_key', 'Unknown'),
                page.get('name', 'Unknown')
            )
            
            if hash_data:
                category_total += hash_data['hash_count']
                
                page_info = {
                    "path": path,
                    "page_name": page.get('name', 'Unknown'),
                    "page_key": page.get('page_key', 'Unknown'),
                    "hash_count": hash_data['hash_count'],
                    "hashes": hash_data['hashes']
                }
                
                category_pages.append(page_info)
                mapping_data["individual_pages"].append(page_info)
                # accumulate for duplicate analysis
                if isinstance(hash_data['hashes'], list):
                    for h in hash_data['hashes']:
                        if isinstance(h, str) and h:
                            aggregated_hashes.append(h)
                            lst = hash_to_pages.get(h)
                            if lst is None:
                                hash_to_pages[h] = [path]
                            else:
                                lst.append(path)
                
                print(f"      ✅ Found {hash_data['hash_count']} files")
            else:
                print(f"      ❌ No files found")
                page_info = {
                    "path": path,
                    "page_name": page.get('name', 'Unknown'),
                    "page_key": page.get('page_key', 'Unknown'),
                    "hash_count": 0,
                    "hashes": []
                }
                category_pages.append(page_info)
                mapping_data["individual_pages"].append(page_info)
        
        # Add category summary (no expected totals)
        mapping_data["categories"][category_name] = {
            "total_pages": len(pages),
            "total_files": category_total,
            "pages": category_pages
        }
        
        total_found += category_total
        
        print(f"   📊 Category {category_name} summary:")
        print(f"      Total pages: {len(pages)}")
        print(f"      Total files: {category_total}")
    
    # Duplicate analysis
    total_entries = len(aggregated_hashes)
    unique_hashes = set(aggregated_hashes)
    total_unique = len(unique_hashes)
    duplicate_count = max(0, total_entries - total_unique)

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 FINAL SUMMARY")
    print(f"   Total individual pages found: {len(mapping_data['individual_pages'])}")
    print(f"   Total files found (page sums): {total_found}")
    print(f"   Unique hashes across all pages: {total_unique}")
    print(f"   Duplicate entries across pages: {duplicate_count}")
    # Print duplicate details
    if duplicate_count > 0:
        print("\nDuplicate hash locations:")
        for h, locs in hash_to_pages.items():
            if isinstance(locs, list) and len(locs) > 1:
                # print hash and first two locations
                print(h)
                print(locs[0])
                print(locs[1] if len(locs) > 1 else "")
    
    
    # Create mappings folder if it doesn't exist
    mappings_dir = "mappings"
    os.makedirs(mappings_dir, exist_ok=True)
    
    # Save the mapping
    timestamp = int(time.time())
    filename = os.path.join(mappings_dir, f"dataset_mapping_{timestamp}.json")
    
    # include duplicate stats in metadata
    mapping_data["metadata"]["total_files_page_sums"] = total_found
    mapping_data["metadata"]["total_unique_hashes"] = total_unique
    mapping_data["metadata"]["duplicate_count"] = duplicate_count
    mapping_data["metadata"]["total_pages"] = len(mapping_data["individual_pages"])
    with open(filename, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\n📁 Mapping saved to: {filename}")

    # Also save a flat hash list for convenience
    try:
        hash_list_path = os.path.join(mappings_dir, f"dataset_hashes_{timestamp}.txt")
        with open(hash_list_path, 'w') as hf:
            for h in sorted(unique_hashes):
                hf.write(h + "\n")
        print(f"📄 Hash list saved to: {hash_list_path} ({len(unique_hashes)} unique)")
    except Exception as e:
        print(f"⚠️  Failed to write hash list: {e}")
    
    return mapping_data

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map pages under a specified hierarchy in Hydrus')
    parser.add_argument('--page', type=str, help='Page hierarchy in format "page1, page2, page3" (defaults to "dataset")')
    args = parser.parse_args()
    
    # Parse page hierarchy
    page_hierarchy = None
    if args.page:
        page_hierarchy = [page.strip() for page in args.page.split(',')]
        print(f"Dataset Structure Mapper (Pages Under: {' -> '.join(page_hierarchy)})")
    else:
        page_hierarchy = ['dataset']
        print("Dataset Structure Mapper (All Pages Under 'Dataset')")
    
    print("=" * 60)
    if page_hierarchy == ['dataset']:
        print("This script will find the top-level 'Dataset' page and recursively")
        print("explore ALL sub-pages, extracting hashes from each individual media page.")
    else:
        print(f"This script will find the page hierarchy {' -> '.join(page_hierarchy)} and recursively")
        print("explore ALL sub-pages, extracting hashes from each individual media page.")
    print("Using corrected logic to properly access the media section.")
    print()
    
    # Get the page structure
    page_data = get_page_structure()
    if not page_data:
        print("❌ Could not retrieve page structure")
        return
    
    # Find all pages under the specified hierarchy
    hierarchy_str = ' -> '.join(page_hierarchy)
    print(f"🔍 Searching for all pages under '{hierarchy_str}'...")
    category_results = find_dataset_hardcore_structure(page_data, page_hierarchy)
    
    if not category_results:
        print(f"❌ Could not find any media pages under '{hierarchy_str}'")
        return
    
    print(f"\n✅ Found {len(category_results)} individual pages:")
    for page, path, category_name in category_results:
        print(f"   - {category_name}: {path}")
    
    # Create the mapping file
    mapping_data = create_mapping_file(category_results)
    
    if mapping_data:
        print(f"\n🎉 SUCCESS! Page hierarchy mapping completed.")
    else:
        print(f"\n❌ Failed to create mapping.")

if __name__ == "__main__":
    main()