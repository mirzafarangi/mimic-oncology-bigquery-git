# Real MIMIC-IV Oncology Pathway Mapping Engine

A dashboard that analyzes **real oncology patient pathways** from the MIMIC-IV dataset using BigQuery.

## 🚨 Quick Fix for Your Setup

Based on your screenshots, here's exactly what you need to do:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Authentication
```bash
# Set your project (from your screenshot)
gcloud config set project mimic-oncology-pathways

# Authenticate with Google Cloud
gcloud auth application-default login
```

### 3. Test Your Setup
```bash
python setup_auth.py
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

## 🔧 Authentication Troubleshooting

### The Issue You're Facing
The **403 Access Denied** error means authentication isn't working properly. Here's how to fix it:

### Option 1: gcloud CLI (Recommended)
```bash
# 1. Make sure you're logged in
gcloud auth list

# 2. Set the correct project
gcloud config set project mimic-oncology-pathways

# 3. Set up application default credentials
gcloud auth application-default login

# 4. Test access
python -c "
from mimic_client import test_connection
print('Success!' if test_connection('mimic-oncology-pathways') else 'Failed!')
"
```

### Option 2: Service Account Key
1. **Download service account key** from Google Cloud Console:
   - Go to IAM & Admin → Service Accounts
   - Create/select a service account
   - Download JSON key

2. **Set environment variable:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
```

### Option 3: Quick Test Script
Create `test_auth.py`:
```python
from google.cloud import bigquery
import os

# Your project ID from screenshot
project_id = "mimic-oncology-pathways"

try:
    client = bigquery.Client(project=project_id)
    
    # Test query to physionet-data (from your screenshot)
    query = """
    SELECT table_name
    FROM `physionet-data.mimiciv_3_1_hosp.__TABLES__`
    WHERE table_name = 'patients'
    LIMIT 1
    """
    
    result = client.query(query).to_dataframe()
    print(f"✅ SUCCESS! Found {len(result)} tables")
    print("Authentication is working correctly!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nTry running:")
    print("gcloud auth application-default login")
```

## 🎯 Your Specific Setup

From your screenshots, I can see:

**Project ID:** `mimic-oncology-pathways`  
**Available Datasets:**
- `physionet-data.mimiciv_3_1_hosp`
- `physionet-data.mimiciv_3_1_icu` 
- `physionet-data.mimiciv_3_1_derived`

The code has been updated to use these exact dataset names.

## 🚀 Step-by-Step Walkthrough

### Step 1: Verify Your Project Access
```bash
# List your projects
gcloud projects list

# Should show: mimic-oncology-pathways
```

### Step 2: Test BigQuery Access
```bash
# Test basic access
bq ls physionet-data:

# Should list the datasets you see in your screenshot
```

### Step 3: Run Authentication Setup
```bash
python setup_auth.py
```

This will:
- ✅ Check your authentication
- ✅ Test access to MIMIC-IV datasets  
- ✅ Run a sample oncology query
- ✅ Save configuration

### Step 4: Launch Dashboard
```bash
streamlit run app.py
```

## 🔍 Common Issues & Solutions

### "403 Forbidden" Error
```bash
# Fix authentication
gcloud auth application-default login

# Verify project access
gcloud config get-value project
```

### "Dataset not found" Error
The code now uses the correct dataset names from your screenshots:
- ✅ `physionet-data.mimiciv_3_1_hosp` (not `mimiciv_hosp`)
- ✅ `physionet-data.mimiciv_3_1_icu` (not `mimiciv_icu`)

### "No oncology patients found"
```python
# Test the oncology query manually
from oncology_extractor import extract_oncology_cohort
patients, events, summary = extract_oncology_cohort("mimic-oncology-pathways", limit=50)
print(f"Found {len(patients)} patients")
```

## 📋 File Structure
```
your-project/
├── app.py                 # Main dashboard (✅ Updated)
├── mimic_client.py        # BigQuery client (✅ Fixed dataset names)
├── oncology_extractor.py  # Data extraction (✅ Updated)
├── setup_auth.py          # Authentication helper (✅ New)
├── requirements.txt       # Dependencies (✅ New)
└── README.md             # This guide (✅ Updated)
```

## ⚡ Quick Commands

```bash
# Full setup in one go
pip install -r requirements.txt
gcloud auth application-default login
gcloud config set project mimic-oncology-pathways
python setup_auth.py
streamlit run app.py
```

## 💡 Pro Tips

1. **Start small:** Use `limit=50` first to test
2. **Check logs:** The dashboard shows detailed error messages
3. **Test connection:** Use the "Test Connection" button in the sidebar
4. **Cache data:** Results are cached for 1 hour to avoid repeated queries

## 🆘 Still Having Issues?

1. **Run the auth setup script:** `python setup_auth.py`
2. **Check the dashboard logs** for specific error messages
3. **Verify MIMIC-IV access** at PhysioNet.org
4. **Test with minimal query** before running full dashboard

---

**Your authentication should work now with the corrected dataset names and proper auth setup!** 🎉