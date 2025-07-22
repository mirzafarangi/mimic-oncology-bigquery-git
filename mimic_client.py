# MIMIC-IV BigQuery Client for Oncology Pathway Analysis
# Real data extraction from MIMIC-IV dataset

import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIMICClient:
    """BigQuery client for MIMIC-IV data extraction."""
    
    def __init__(self, project_id: str = None):
        """Initialize BigQuery client."""
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("Project ID must be provided or set in GOOGLE_CLOUD_PROJECT environment variable")
        
        try:
            # Initialize BigQuery client with explicit project
            self.client = bigquery.Client(project=self.project_id)
            
            # Use the correct MIMIC-IV v3.1 dataset references from your screenshots
            self.mimic_dataset = 'physionet-data.mimiciv_3_1_hosp'
            self.mimic_icu = 'physionet-data.mimiciv_3_1_icu'
            self.mimic_derived = 'physionet-data.mimiciv_3_1_derived'
            
            logger.info(f"Initialized MIMIC client for project: {self.project_id}")
            logger.info(f"Using dataset: {self.mimic_dataset}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test BigQuery connection and MIMIC-IV access."""
        try:
            logger.info(f"Testing connection to project: {self.project_id}")
            logger.info(f"Testing access to dataset: {self.mimic_dataset}")
            
            # Test direct table access (most reliable)
            logger.info("Executing test query...")
            simple_query = f"SELECT COUNT(*) as count FROM `{self.mimic_dataset}.patients` LIMIT 1"
            result = self.client.query(simple_query).to_dataframe()
            
            if len(result) > 0:
                count = result['count'].iloc[0]
                logger.info(f"✅ Connection successful. Found {count:,} patients")
                return True
            else:
                logger.error("❌ No results from test query - check dataset access")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            logger.error(f"Make sure you have:")
            logger.error(f"1. Proper authentication set up (gcloud auth or service account)")
            logger.error(f"2. Access to physionet-data datasets")
            logger.error(f"3. Your project ID is correct: {self.project_id}")
            return False
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from any MIMIC table."""
        try:
            query = f"SELECT * FROM `{self.mimic_dataset}.{table_name}` LIMIT {limit}"
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Error sampling {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def explore_dataset(self) -> pd.DataFrame:
        """Get overview of MIMIC-IV dataset structure."""
        try:
            query = f"""
            SELECT 
                table_id as table_name,
                row_count
            FROM `{self.mimic_dataset}.__TABLES__`
            WHERE table_type = 'BASE_TABLE'
            ORDER BY table_id
            """
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Error exploring dataset: {str(e)}")
            return pd.DataFrame()

def test_connection(project_id: str = None) -> bool:
    """Standalone function to test MIMIC-IV connection."""
    try:
        client = MIMICClient(project_id)
        return client.test_connection()
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

# ICD Code Mappings for Oncology (Updated and Comprehensive)
CANCER_ICD_MAPPINGS = {
    # Hematologic Malignancies
    'hodgkin_lymphoma': {
        'icd10': ['C81.0', 'C81.1', 'C81.2', 'C81.3', 'C81.4', 'C81.7', 'C81.9'],
        'icd9': ['201.0', '201.1', '201.2', '201.4', '201.7', '201.9'],
        'category': 'Hematologic',
        'name': 'Hodgkin Lymphoma'
    },
    'non_hodgkin_lymphoma': {
        'icd10': ['C82.0', 'C82.1', 'C82.2', 'C82.3', 'C82.4', 'C82.5', 'C82.6', 'C82.7', 'C82.8', 'C82.9',
                  'C83.0', 'C83.1', 'C83.3', 'C83.5', 'C83.7', 'C83.8', 'C83.9',
                  'C84.0', 'C84.1', 'C84.4', 'C84.6', 'C84.7', 'C84.8', 'C84.9',
                  'C85.1', 'C85.2', 'C85.7', 'C85.8', 'C85.9'],
        'icd9': ['200.0', '200.1', '200.2', '200.8', '202.0', '202.1', '202.2', '202.8', '202.9'],
        'category': 'Hematologic',
        'name': 'Non-Hodgkin Lymphoma'
    },
    'multiple_myeloma': {
        'icd10': ['C90.0', 'C90.1', 'C90.2'],
        'icd9': ['203.0', '203.1'],
        'category': 'Hematologic',
        'name': 'Multiple Myeloma'
    },
    'acute_leukemia': {
        'icd10': ['C91.0', 'C92.0', 'C93.0', 'C94.0', 'C95.0'],
        'icd9': ['204.0', '205.0', '206.0', '207.0', '208.0'],
        'category': 'Hematologic',
        'name': 'Acute Leukemia'
    },
    'chronic_leukemia': {
        'icd10': ['C91.1', 'C92.1', 'C93.1', 'C94.1', 'C95.1'],
        'icd9': ['204.1', '205.1', '206.1', '207.1', '208.1'],
        'category': 'Hematologic',
        'name': 'Chronic Leukemia'
    },
    
    # Thyroid Cancers
    'thyroid_cancer': {
        'icd10': ['C73'],
        'icd9': ['193'],
        'category': 'Thyroid',
        'name': 'Thyroid Cancer'
    },
    
    # Gastrointestinal Cancers
    'colorectal_cancer': {
        'icd10': ['C18.0', 'C18.1', 'C18.2', 'C18.3', 'C18.4', 'C18.5', 'C18.6', 'C18.7', 'C18.8', 'C18.9',
                  'C19', 'C20', 'C21.0', 'C21.1', 'C21.2', 'C21.8'],
        'icd9': ['153.0', '153.1', '153.2', '153.3', '153.4', '153.5', '153.6', '153.7', '153.8', '153.9',
                 '154.0', '154.1', '154.2', '154.3', '154.8'],
        'category': 'Gastrointestinal',
        'name': 'Colorectal Cancer'
    },
    'gastric_cancer': {
        'icd10': ['C16.0', 'C16.1', 'C16.2', 'C16.3', 'C16.4', 'C16.5', 'C16.6', 'C16.8', 'C16.9'],
        'icd9': ['151.0', '151.1', '151.2', '151.3', '151.4', '151.5', '151.6', '151.8', '151.9'],
        'category': 'Gastrointestinal',
        'name': 'Gastric Cancer'
    },
    'pancreatic_cancer': {
        'icd10': ['C25.0', 'C25.1', 'C25.2', 'C25.3', 'C25.4', 'C25.7', 'C25.8', 'C25.9'],
        'icd9': ['157.0', '157.1', '157.2', '157.3', '157.4', '157.8', '157.9'],
        'category': 'Gastrointestinal',
        'name': 'Pancreatic Cancer'
    },
    'hepatocellular_carcinoma': {
        'icd10': ['C22.0', 'C22.1'],
        'icd9': ['155.0', '155.1'],
        'category': 'Gastrointestinal',
        'name': 'Hepatocellular Carcinoma'
    },
    'esophageal_cancer': {
        'icd10': ['C15.0', 'C15.1', 'C15.2', 'C15.3', 'C15.4', 'C15.5', 'C15.8', 'C15.9'],
        'icd9': ['150.0', '150.1', '150.2', '150.3', '150.4', '150.5', '150.8', '150.9'],
        'category': 'Gastrointestinal',
        'name': 'Esophageal Cancer'
    },
    
    # Lung Cancers
    'lung_cancer': {
        'icd10': ['C34.0', 'C34.1', 'C34.2', 'C34.3', 'C34.8', 'C34.9'],
        'icd9': ['162.0', '162.1', '162.2', '162.3', '162.4', '162.5', '162.8', '162.9'],
        'category': 'Thoracic',
        'name': 'Lung Cancer'
    },
    
    # Genitourinary Cancers
    'prostate_cancer': {
        'icd10': ['C61'],
        'icd9': ['185'],
        'category': 'Genitourinary',
        'name': 'Prostate Cancer'
    },
    'renal_cell_carcinoma': {
        'icd10': ['C64'],
        'icd9': ['189.0'],
        'category': 'Genitourinary',
        'name': 'Renal Cell Carcinoma'
    },
    'bladder_cancer': {
        'icd10': ['C67.0', 'C67.1', 'C67.2', 'C67.3', 'C67.4', 'C67.5', 'C67.6', 'C67.7', 'C67.8', 'C67.9'],
        'icd9': ['188.0', '188.1', '188.2', '188.3', '188.4', '188.5', '188.6', '188.7', '188.8', '188.9'],
        'category': 'Genitourinary',
        'name': 'Bladder Cancer'
    },
    
    # Gynecologic Cancers
    'ovarian_cancer': {
        'icd10': ['C56'],
        'icd9': ['183.0'],
        'category': 'Gynecologic',
        'name': 'Ovarian Cancer'
    },
    'endometrial_cancer': {
        'icd10': ['C54.0', 'C54.1', 'C54.2', 'C54.3', 'C54.8', 'C54.9'],
        'icd9': ['182.0', '182.1', '182.8'],
        'category': 'Gynecologic',
        'name': 'Endometrial Cancer'
    },
    'cervical_cancer': {
        'icd10': ['C53.0', 'C53.1', 'C53.8', 'C53.9'],
        'icd9': ['180.0', '180.1', '180.8', '180.9'],
        'category': 'Gynecologic',
        'name': 'Cervical Cancer'
    },
    
    # Breast Cancer
    'breast_cancer': {
        'icd10': ['C50.0', 'C50.1', 'C50.2', 'C50.3', 'C50.4', 'C50.5', 'C50.6', 'C50.8', 'C50.9'],
        'icd9': ['174.0', '174.1', '174.2', '174.3', '174.4', '174.5', '174.6', '174.8', '174.9', '175.0', '175.9'],
        'category': 'Breast',
        'name': 'Breast Cancer'
    },
    
    # Brain Tumors
    'brain_cancer': {
        'icd10': ['C71.0', 'C71.1', 'C71.2', 'C71.3', 'C71.4', 'C71.5', 'C71.6', 'C71.7', 'C71.8', 'C71.9'],
        'icd9': ['191.0', '191.1', '191.2', '191.3', '191.4', '191.5', '191.6', '191.7', '191.8', '191.9'],
        'category': 'Central Nervous System',
        'name': 'Brain Cancer'
    },
    
    # Dermatologic
    'melanoma': {
        'icd10': ['C43.0', 'C43.1', 'C43.2', 'C43.3', 'C43.4', 'C43.5', 'C43.6', 'C43.7', 'C43.8', 'C43.9'],
        'icd9': ['172.0', '172.1', '172.2', '172.3', '172.4', '172.5', '172.6', '172.7', '172.8', '172.9'],
        'category': 'Dermatologic',
        'name': 'Melanoma'
    },
    
    # General cancer codes
    'malignant_neoplasm': {
        'icd10': ['C78.1', 'C80.1'],  # Secondary malignant neoplasm and malignant neoplasm unspecified
        'icd9': ['196.0', '196.1', '196.2', '196.3', '196.5', '196.6', '196.8', '196.9', '199.0', '199.1'],
        'category': 'Other',
        'name': 'Malignant Neoplasm'
    }
}

def get_cancer_icd_codes() -> Dict[str, List[str]]:
    """Get all ICD codes for cancer detection."""
    all_codes = []
    for cancer_data in CANCER_ICD_MAPPINGS.values():
        all_codes.extend(cancer_data['icd10'])
        all_codes.extend(cancer_data['icd9'])
    return list(set(all_codes))

def map_icd_to_cancer_type(icd_code: str) -> Optional[str]:
    """Map an ICD code to a cancer type."""
    for cancer_key, cancer_data in CANCER_ICD_MAPPINGS.items():
        if icd_code in cancer_data['icd10'] or icd_code in cancer_data['icd9']:
            return cancer_data['name']
    return None

def get_cancer_category(cancer_type: str) -> str:
    """Get cancer category from cancer type."""
    for cancer_data in CANCER_ICD_MAPPINGS.values():
        if cancer_data['name'] == cancer_type:
            return cancer_data['category']
    return 'Other'