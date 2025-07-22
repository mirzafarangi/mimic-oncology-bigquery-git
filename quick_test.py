#!/usr/bin/env python3
"""
Quick test to verify everything is working
Run this before launching the dashboard
"""

import sys
import logging
from mimic_client import test_connection
from oncology_extractor import extract_oncology_cohort

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def main():
    project_id = "mimic-oncology-pathways"
    
    print("="*60)
    print("🏥 Quick MIMIC-IV Test")
    print("="*60)
    
    print("\n1. Testing BigQuery connection...")
    if test_connection(project_id):
        print("✅ Connection successful!")
    else:
        print("❌ Connection failed!")
        sys.exit(1)
    
    print("\n2. Testing oncology data extraction...")
    try:
        patients_df, events_df, summary = extract_oncology_cohort(project_id, limit=5)  # Very small test
        
        print(f"✅ Found {len(patients_df)} oncology patients")
        print(f"✅ Generated {len(events_df)} clinical events")
        
        if len(patients_df) > 0:
            print(f"\nSample cancer types found:")
            cancer_counts = patients_df['cancer_type'].value_counts()
            for cancer, count in cancer_counts.head(5).items():
                print(f"  • {cancer}: {count}")
        
        if len(events_df) > 0:
            print(f"\nSample event types:")
            event_counts = events_df['event_type'].value_counts()
            for event_type, count in event_counts.items():
                print(f"  • {event_type}: {count}")
        else:
            print("⚠️  No clinical events found (this is OK for testing)")
        
        print(f"\n🎉 SUCCESS! Everything is working.")
        print(f"🚀 Ready to run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        print("\n🔍 To debug schema issues, run:")
        print("   python check_schema.py")
        sys.exit(1)

if __name__ == "__main__":
    main()