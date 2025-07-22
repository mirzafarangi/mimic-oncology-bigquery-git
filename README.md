# MIMIC-IV Oncology Pathway Mapping Engine

A  dashboard that analyzes **real oncology patient pathways** from the MIMIC-IV dataset using BigQuery.



Your infinite loading issue was caused by:
1. **Streamlit caching conflicts** - Fixed with session state management
2. **SQL column name errors** - Fixed with correct MIMIC-IV v3.1 schema  
3. **Connection testing loops** - Fixed with simplified authentication

## 🚀 **SETUP**

### 1. Files
Make sure you have these files in your project directory:
- ✅ `app.py` 
- ✅ `mimic_client.py` 
- ✅ `oncology_extractor.py` 
- ✅ `requirements.txt` 
- ✅ `setup_auth.py` (Authentication helper)
- ✅ `quick_test.py` (Testing script)
- ✅ `check_schema.py` (Schema checker)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Authentication 
```bash
gcloud config set project mimic-oncology-pathways
gcloud auth application-default login
```


### 4. Test Everything Works
```bash
python quick_test.py
```

Should output:
```
✅ Connection successful. Found 364,627 patients
✅ Found X oncology patients  
✅ Generated X clinical events
🎉 SUCCESS! Everything is working.
```

### 5. Run the Dashboard
```bash
streamlit run app.py
```



## 📊 ** Features**

### 🎯 **Smart Data Loading:**
- Click "Load MIMIC-IV Data" button in sidebar
- Choose number of patients (start with 10-20)
- Real-time progress indicators
- Clear error messages if something goes wrong
- Session state prevents re-queries
- Manual data refresh control
- Cache clearing button
- Connection testing separate from data loading

### 📈 **Analytics:**
- **Real patient demographics** from MIMIC-IV
- **Actual cancer diagnoses** based on ICD codes  
- **Clinical pathways** from real hospital data
- **Treatment patterns** and outcomes
- **Digital twin matching** with real patients

## 🎯 **How It Works **

1. **Enter your project ID** in the sidebar (`mimic-oncology-pathways`)
2. **Set patient limit** (start with 10-50 for testing)
3. **Click "Load MIMIC-IV Data"** button
4. **Wait for progress messages** (no infinite spinner!)
5. **Explore the dashboard** with real data

## 🔍 **Troubleshooting Guide**

### If You Get Errors:

**Connection Failed:**
```bash
# Re-authenticate
gcloud auth application-default login
```

**No Patients Found:**
```bash
# Test with larger limit
# In sidebar, increase "Max Patients to Load" to 50-100
```

**Schema Errors:**
```bash
# Check table structure
python check_schema.py
```

**Still Having Issues:**
```bash
# Run full diagnostic
python setup_auth.py
```

## 📋 **Cancer Types Detected**

The system now finds these cancer types from real MIMIC-IV data:

### **Hematologic Malignancies:**
- Hodgkin Lymphoma
- Non-Hodgkin Lymphoma  
- Multiple Myeloma
- Acute/Chronic Leukemias

### **Solid Tumors:**
- **Thyroid Cancer** (all subtypes)
- **Gastrointestinal:** Colorectal, Gastric, Pancreatic, Hepatocellular, Esophageal
- **Thoracic:** Lung Cancer (all types)
- **Genitourinary:** Prostate, Renal Cell, Bladder
- **Gynecologic:** Ovarian, Endometrial, Cervical
- **Breast Cancer**
- **Brain Cancer**  
- **Melanoma**

## ⚡ **Performance Optimized**

- **Start small:** Load 10-20 patients first
- **Session caching:** Data persists until you reload
- **Smart filtering:** Apply filters without re-querying
- **Export capability:** Download filtered results

## 🎉 **Success Indicators**

When working correctly, you'll see:

### **In Terminal:**
```
✅ Connection successful. Found 364,627 patients
INFO:oncology_extractor:🎯 Extracting oncology cohort (limit: 10)
INFO:oncology_extractor:Found 10 oncology patients  
INFO:oncology_extractor:🔄 Building clinical pathways...
INFO:oncology_extractor:Generated 45 clinical events
✅ Successfully loaded 10 patients with 45 events
```

### **In Dashboard:**
- ✅ Green success messages
- 📊 Real patient data in tables
- 🎯 Actual cancer types from MIMIC-IV  
- 📈 Interactive charts and visualizations
- 👥 Digital twin matching with real patients

## 🔒 **Data Privacy & Ethics**

- ✅ **All data remains in BigQuery** - no PHI downloaded
- ✅ **Uses anonymized MIMIC-IV data** only
- ✅ **Follows PhysioNet data use agreements**
- ✅ **Research and educational use only**


## 🎯 **Quick Start Commands**

```bash
# Test everything works
python quick_test.py

# Run the dashboard  
streamlit run app.py

# Open in browser: http://localhost:8501
# 1. Click "Load MIMIC-IV Data" 
# 2. Wait for success message
# 3. Explore real oncology pathways!
```

---
