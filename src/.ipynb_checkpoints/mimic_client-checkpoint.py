"""
Simplified MIMIC-IV BigQuery client.
"""

import yaml
from google.cloud import bigquery
import pandas as pd
from typing import Dict, List, Optional
import warnings
import os

# Suppress BigQuery Storage warning
warnings.filterwarnings('ignore', message='BigQuery Storage module not found')

class MIMICClient:
    """Simple client for MIMIC-IV BigQuery access."""
    
    def __init__(self, config_path: str = None):
        """Initialize with your billing project."""
        if config_path is None:
            # Auto-detect config path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), "config", "project_config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.client = bigquery.Client(project=self.config['billing_project'])
        self.hospital_dataset = self.config['mimic_datasets']['hospital']
        
    def query(self, sql: str) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        try:
            result = self.client.query(sql).to_dataframe()
            print(f"✅ Query executed: {len(result):,} rows returned")
            return result
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            print("💡 Check your SQL syntax and table names")
            raise
    
    def get_table_sample(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Get sample data from a table."""
        query = f"""
        SELECT * 
        FROM `{self.hospital_dataset}.{table_name}`
        LIMIT {limit}
        """
        return self.query(query)
    
    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table."""
        query = f"""
        SELECT COUNT(*) as total
        FROM `{self.hospital_dataset}.{table_name}`
        """
        result = self.query(query)
        return result['total'].iloc[0]
    
    def explore_dataset(self) -> pd.DataFrame:
        """Get overview of key tables."""
        
        # Get counts for main tables
        tables_to_check = ['patients', 'admissions', 'diagnoses_icd', 'procedures_icd']
        
        table_info = []
        for table in tables_to_check:
            try:
                count = self.get_table_count(table)
                table_info.append({
                    'table_name': table,
                    'row_count': f"{count:,}",
                    'status': '✅ Available'
                })
            except Exception as e:
                table_info.append({
                    'table_name': table,
                    'row_count': 'N/A',
                    'status': f'❌ Error: {str(e)[:50]}'
                })
        
        return pd.DataFrame(table_info)


def test_connection() -> bool:
    """Test connection to MIMIC-IV."""
    try:
        client = MIMICClient()
        
        # Simple test query
        test_result = client.query("""
        SELECT 'connection_test' as test, COUNT(*) as patient_count
        FROM `physionet-data.mimiciv_3_1_hosp.patients`
        """)
        
        if len(test_result) > 0:
            print(f"✅ Connection successful! Found {test_result['patient_count'].iloc[0]:,} patients")
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False