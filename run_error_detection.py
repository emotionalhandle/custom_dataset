#!/usr/bin/env python3
"""
run_error_detection.py - Master script to run all error detection methods
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n🔍 {description}")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print("Error:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_required_files() -> List[str]:
    """Check if required files exist."""
    required_files = [
        'dataset_hardcore_mapper.py',
        'advanced_joytag_processing.py',
        'find_mistagged_images.py',
        'find_near_duplicates.py',
        'validate_series_predictions.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def find_latest_dataset() -> str:
    """Find the most recent dataset file."""
    dataset_files = []
    for file in os.listdir('.'):
        if file.startswith('advanced_joytag_complete_') and file.endswith('.json'):
            dataset_files.append(file)
    
    if not dataset_files:
        return None
    
    return sorted(dataset_files)[-1]

def generate_summary_report(results: Dict[str, bool]) -> str:
    """Generate a summary report of all error detection results."""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_checks': len(results),
            'successful_checks': sum(results.values()),
            'failed_checks': len(results) - sum(results.values())
        },
        'results': results,
        'recommendations': []
    }
    
    # Add recommendations based on results
    if results.get('mistagged_detection', False):
        report['recommendations'].append("Review mistagged images for missing or incorrect tags")
    
    if results.get('duplicate_detection', False):
        report['recommendations'].append("Check near-duplicate images for inconsistent tagging")
    
    if results.get('validation_detection', False):
        report['recommendations'].append("Verify series classifier predictions against existing tags")
    
    # Save report
    report_file = f"error_detection_summary_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    return report_file

def main():
    print("🔍 DATASET ERROR DETECTION WORKFLOW")
    print("=" * 50)
    print("This script will run all error detection methods on your dataset")
    print()
    
    # Check required files
    missing_files = check_required_files()
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all error detection scripts are present")
        return
    
    # Check if dataset exists
    dataset_file = find_latest_dataset()
    if not dataset_file:
        print("❌ No processed dataset found")
        print("Please run 'advanced_joytag_processing.py' first")
        return
    
    print(f"📁 Using dataset: {dataset_file}")
    print()
    
    # Run error detection methods
    results = {}
    
    # 1. Mistagged images detection
    results['mistagged_detection'] = run_script(
        'find_mistagged_images.py',
        'Detecting mistagged images'
    )
    
    # 2. Near duplicate detection
    results['duplicate_detection'] = run_script(
        'find_near_duplicates.py',
        'Finding near-duplicate images'
    )
    
    # 3. Series prediction validation
    results['validation_detection'] = run_script(
        'validate_series_predictions.py',
        'Validating series predictions'
    )
    
    # Generate summary report
    report_file = generate_summary_report(results)
    
    # Print final summary
    print("\n📊 ERROR DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total checks: {len(results)}")
    print(f"Successful: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")
    
    print("\n📁 Generated reports:")
    for check, success in results.items():
        if success:
            print(f"   ✅ {check.replace('_', ' ').title()}")
        else:
            print(f"   ❌ {check.replace('_', ' ').title()}")
    
    print(f"\n📄 Summary report: {report_file}")
    
    # Show recommendations
    if sum(results.values()) > 0:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Review the generated error reports")
        print("2. Use the web dashboard to manually review errors")
        print("3. Fix identified tagging issues")
        print("4. Re-run error detection to verify fixes")
    
    print("\n🌐 To view errors interactively, run:")
    print("   python webapp/app.py")
    print("   Then visit: http://localhost:5000/error-review")

if __name__ == '__main__':
    main()