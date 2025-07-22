#!/usr/bin/env python3
"""
Authentication Setup for MIMIC-IV BigQuery Access
Run this script to test and configure your authentication
"""

import os
import sys
import json
from pathlib import Path
from google.cloud import bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

def check_gcloud_auth():
    """Check if gcloud authentication is set up."""
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ gcloud CLI is installed and configured")
            print("Active accounts:")
            print(result.stdout)
            return True
        else:
            print("❌ gcloud auth list failed")
            return False
    except FileNotFoundError:
        print("❌ gcloud CLI not found. Please install Google Cloud CLI")
        return False

def check_credentials():
    """Check various credential sources."""
    print("\n🔍 Checking authentication methods...\n")
    
    # Check 1: Environment variable
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS set to: {cred_path}")
        if os.path.exists(cred_path):
            print("✅ Service account file exists")
            try:
                with open(cred_path) as f:
                    cred_data = json.load(f)
                    print(f"✅ Service account: {cred_data.get('client_email', 'Unknown')}")
            except Exception as e:
                print(f"❌ Error reading service account file: {e}")
        else:
            print("❌ Service account file does not exist!")
    else:
        print("ℹ️  GOOGLE_APPLICATION_CREDENTIALS not set")
    
    # Check 2: Default credentials
    try:
        credentials, project_id = default()
        print(f"✅ Default credentials found. Project: {project_id}")
        return credentials, project_id
    except DefaultCredentialsError as e:
        print(f"❌ No default credentials found: {e}")
        return None, None

def test_bigquery_access(project_id):
    """Test BigQuery access with different datasets."""
    print(f"\n🧪 Testing BigQuery access for project: {project_id}\n")
    
    try:
        client = bigquery.Client(project=project_id)
        
        # Test datasets that exist in physionet-data
        test_datasets = [
            'physionet-data.mimiciv_3_1_hosp',
            'physionet-data.mimiciv_3_1_icu',
            'physionet-data.mimiciv_3_1_derived'
        ]
        
        for dataset_name in test_datasets:
            try:
                print(f"Testing access to {dataset_name}...")
                
                # List tables in the dataset
                query = f"""
                SELECT table_id, row_count
                FROM `{dataset_name}.__TABLES__`
                WHERE table_id IN ('patients', 'admissions', 'diagnoses_icd')
                ORDER BY table_id
                LIMIT 5
                """
                
                result = client.query(query).to_dataframe()
                
                if len(result) > 0:
                    print(f"✅ {dataset_name} - Found {len(result)} tables:")
                    for _, row in result.iterrows():
                        row_count = row['row_count'] if 'row_count' in result.columns else 'Unknown'
                        print(f"  • {row['table_id']}: {row_count:,} rows")
                else:
                    print(f"⚠️  {dataset_name} - No tables found or no access")
                    
            except Exception as e:
                print(f"❌ {dataset_name} - Access failed: {str(e)}")
                
        print("\n" + "="*60)
        
        # Test oncology patient query
        print("\n🎯 Testing oncology patient extraction...\n")
        
        oncology_query = f"""
        SELECT 
            COUNT(DISTINCT d.subject_id) as oncology_patients,
            COUNT(d.icd_code) as oncology_diagnoses
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
        WHERE d.icd_code IN ('C78.1', 'C80.1', '185', '193', '151.9', '153.9', '157.9', '162.9', '174.9')
        """
        
        try:
            result = client.query(oncology_query).to_dataframe()
            if len(result) > 0:
                print(f"✅ Found {result['oncology_patients'].iloc[0]} potential oncology patients")
                print(f"✅ Found {result['oncology_diagnoses'].iloc[0]} oncology diagnoses")
                return True
            else:
                print("⚠️  No oncology patients found")
                return False
        except Exception as e:
            print(f"❌ Oncology query failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ BigQuery client initialization failed: {str(e)}")
        return False

def setup_authentication():
    """Interactive authentication setup."""
    print("\n🔧 Authentication Setup Options:\n")
    print("1. Use gcloud CLI authentication (recommended for development)")
    print("2. Use service account key file")
    print("3. Skip setup (authentication already configured)")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        print("\n📋 Setting up gcloud CLI authentication:")
        print("\n1. Install Google Cloud CLI if not already installed:")
        print("   https://cloud.google.com/sdk/docs/install")
        print("\n2. Run the following commands:")
        print(f"   gcloud config set project mimic-oncology-pathways")
        print("   gcloud auth application-default login")
        print("\n3. Follow the browser authentication flow")
        
    elif choice == "2":
        print("\n📋 Setting up service account authentication:")
        print("\n1. Go to Google Cloud Console")
        print("2. Navigate to IAM & Admin > Service Accounts")
        print("3. Create a new service account or use existing one")
        print("4. Download the JSON key file")
        print("5. Set environment variable:")
        print("   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/key.json'")
        
    elif choice == "3":
        print("✅ Skipping setup - assuming authentication is configured")
        
    else:
        print("❌ Invalid choice")

def main():
    """Main setup function."""
    print("="*60)
    print("🏥 MIMIC-IV BigQuery Authentication Setup")
    print("="*60)
    
    # Check current status
    credentials, project_id = check_credentials()
    gcloud_ok = check_gcloud_auth()
    
    # Use provided project ID or detect from environment
    if not project_id:
        project_id = input("\n📝 Enter your Google Cloud Project ID (mimic-oncology-pathways): ").strip()
        if not project_id:
            project_id = "mimic-oncology-pathways"
    
    print(f"\n🎯 Using project ID: {project_id}")
    
    # Test BigQuery access
    if credentials or gcloud_ok:
        if test_bigquery_access(project_id):
            print("\n🎉 SUCCESS! Your authentication is working correctly.")
            print("\n🚀 You can now run: streamlit run app.py")
            
            # Save configuration
            config = {
                'project_id': project_id,
                'dataset': 'physionet-data.mimiciv_3_1_hosp'
            }
            
            with open('.mimic_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Configuration saved to .mimic_config.json")
            
        else:
            print("\n❌ Authentication test failed.")
            setup_authentication()
    else:
        print("\n❌ No authentication found.")
        setup_authentication()

if __name__ == "__main__":
    main()