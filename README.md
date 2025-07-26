# MIMIC-IV Oncology Pathway Mapping Engine

A clinical pathway analysis platform that analyzes real oncology patient data from the MIMIC-IV dataset using Google BigQuery.


## 🎯 Key Features

- **Real Patient Data Analysis**: Processes actual oncology cohorts from MIMIC-IV
- **Clinical Pathway Visualization**: Interactive Sankey diagrams and timeline views
- **Digital Twin Matching**: Find similar patients based on demographics and diagnosis
- **Survival Analysis**: Kaplan-Meier curves with stratification options
- **BigQuery Integration**: Scalable processing of large medical datasets
- **HIPAA Compliant**: All data remains in secure BigQuery environment

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Cloud account with BigQuery access
- MIMIC-IV dataset access via PhysioNet

### Installation

```bash
# Clone the repository
git clone https://github.com/mirzafarangi/mimic-oncology-bigquery-git.git
cd mimic-oncology-bigquery-git

# Install dependencies
pip install -r requirements.txt

# Configure Google Cloud
gcloud auth application-default login
gcloud config set project your-project-id
```

### Running the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 📊 Cancer Types Analyzed

### Hematologic Malignancies
- Hodgkin & Non-Hodgkin Lymphoma
- Multiple Myeloma
- Acute & Chronic Leukemias

### Solid Tumors
- **Thyroid**: Papillary, Follicular, Medullary, Anaplastic
- **Gastrointestinal**: Colorectal, Gastric, Pancreatic, Hepatocellular
- **Thoracic**: Lung (NSCLC, SCLC)
- **Genitourinary**: Prostate, Renal Cell, Bladder
- **Gynecologic**: Ovarian, Endometrial, Cervical
- **Others**: Breast, Brain, Melanoma

## 🔧 Configuration

Create a `.env` file in the project root:

```env
PROJECT_ID=your-bigquery-project
DATASET_ID=mimiciv_hosp
MAX_PATIENTS=1000
CACHE_TTL=3600
```

## 📈 Usage Examples

### Loading Patient Data

```python
from mimic_client import MIMICClient

# Initialize client
client = MIMICClient(project_id='your-project')

# Extract oncology cohort
oncology_data = client.extract_oncology_cohort(limit=100)
patients_df = oncology_data['patients']
events_df = oncology_data['events']
```

### Digital Twin Matching

```python
# Find similar patients
similar_patients = find_digital_twins(
    age=65,
    gender='M',
    cancer_type='Colorectal Cancer',
    stage='T3N1M0'
)
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│ BigQuery API │────▶│  MIMIC-IV Data  │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────┐
│  Visualizations │     │   Analytics  │
└─────────────────┘     └──────────────┘
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.

# Quick connection test
python quick_test.py
```

## 📝 Documentation

- [API Reference](docs/api.md)
- [Data Schema](docs/schema.md)
- [User Guide](docs/user_guide.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance

- Processes 1000 patients in ~5 seconds
- Supports cohorts up to 10,000 patients
- Real-time filtering and visualization
- Cached queries for improved performance

## 🔒 Security & Privacy

- No PHI downloaded to local machine
- All processing done in BigQuery
- Compliant with MIMIC-IV data use agreement
- Session-based authentication

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{mimic_oncology_pathways,
  author = {Mirzafarangi},
  title = {MIMIC-IV Oncology Pathway Mapping Engine},
  year = {2024},
  url = {https://github.com/mirzafarangi/mimic-oncology-bigquery-git}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MIMIC-IV dataset and PhysioNet team
- Streamlit for the amazing framework
- Google BigQuery for scalable data processing

---


<!-- Update: 2025-07-23 14:18:44.644452 -->

<!-- Update: 2025-07-23 14:18:44.652111 -->

<!-- Update: 2025-07-23 14:18:44.658370 -->
