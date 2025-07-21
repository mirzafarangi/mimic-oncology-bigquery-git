#!/usr/bin/env python3

import sys
sys.path.append('src')

print("🔍 Testing MIMIC-IV setup...")

try:
    from mimic_client import MIMICClient, test_connection
    print("✅ Imports successful")
    
    # Test connection
    if test_connection():
        print("✅ BigQuery connection successful")
        
        # Test client initialization
        client = MIMICClient()
        print("✅ Client initialized successfully")
        print(f"✅ Using billing project: {client.config['billing_project']}")
        
    else:
        print("❌ Connection test failed")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()