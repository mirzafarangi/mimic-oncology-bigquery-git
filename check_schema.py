#!/usr/bin/env python3
"""
Check MIMIC-IV table schemas to verify column names
"""

from google.cloud import bigquery

def check_table_schema(project_id="mimic-oncology-pathways"):
    """Check the schema of key MIMIC-IV tables."""
    
    client = bigquery.Client(project=project_id)
    
    tables_to_check = [
        'physionet-data.mimiciv_3_1_hosp.prescriptions',
        'physionet-data.mimiciv_3_1_hosp.procedures_icd',
        'physionet-data.mimiciv_3_1_hosp.diagnoses_icd',
        'physionet-data.mimiciv_3_1_hosp.patients'
    ]
    
    for table_name in tables_to_check:
        print(f"\n{'='*60}")
        print(f"📋 {table_name}")
        print(f"{'='*60}")
        
        try:
            # Get table schema
            table = client.get_table(table_name)
            
            print(f"Columns ({len(table.schema)}):")
            for field in table.schema:
                print(f"  • {field.name} ({field.field_type})")
            
            # Get sample data
            query = f"SELECT * FROM `{table_name}` LIMIT 3"
            result = client.query(query).to_dataframe()
            
            if len(result) > 0:
                print(f"\nSample data:")
                print(result.head().to_string())
            
        except Exception as e:
            print(f"❌ Error accessing {table_name}: {e}")

def main():
    print("🔍 Checking MIMIC-IV Table Schemas")
    check_table_schema()

if __name__ == "__main__":
    main()