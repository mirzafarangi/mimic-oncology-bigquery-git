# =================================================================
# COMPREHENSIVE ONCOLOGY CANCER REGISTRY
# =================================================================

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# =================================================================
# 1. CANCER CONFIGURATION SYSTEM
# =================================================================

@dataclass
class CancerConfig:
    """Configuration class for cancer types."""
    name: str
    category: str
    incidence_rate: float  # Proportion in dataset
    age_distribution: Dict[str, float]  # mean, std
    gender_distribution: Dict[str, float]  # M, F probabilities
    staging_system: str  # TNM, Ann_Arbor, etc.
    stages: List[str]
    stage_probabilities: List[float]
    treatments: Dict[str, float]  # Treatment: probability
    complications: Dict[str, float]  # Complication: probability
    outcomes: Dict[str, float]  # Outcome: probability
    outcome_modifiers: Dict[str, Dict[str, float]]  # Stage-specific outcomes
    timeline_days: Dict[str, Tuple[int, int]]  # Event: (min_days, max_days)

# =================================================================
# 2. COMPREHENSIVE CANCER REGISTRY (25+ Cancer Types)
# =================================================================

CANCER_REGISTRY = {
    # HEMATOLOGIC MALIGNANCIES
    'hodgkin_lymphoma': CancerConfig(
        name='Hodgkin Lymphoma',
        category='Hematologic',
        incidence_rate=0.025,
        age_distribution={'mean': 35, 'std': 15},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='Ann_Arbor',
        stages=['I', 'II', 'III', 'IV'],
        stage_probabilities=[0.3, 0.3, 0.25, 0.15],
        treatments={
            'ABVD': 0.6,
            'BEACOPP': 0.2,
            'Radiation Therapy': 0.4,
            'Stem Cell Transplant': 0.1
        },
        complications={
            'Neutropenia': 0.4,
            'Pulmonary Toxicity': 0.15,
            'Secondary Malignancy': 0.05
        },
        outcomes={
            'Complete Remission': 0.85,
            'Relapse': 0.12,
            'Death': 0.03
        },
        outcome_modifiers={
            'I': {'Complete Remission': 0.95, 'Relapse': 0.04, 'Death': 0.01},
            'IV': {'Complete Remission': 0.70, 'Relapse': 0.20, 'Death': 0.10}
        },
        timeline_days={
            'diagnosis_to_treatment': (7, 21),
            'treatment_cycle': (21, 28),
            'follow_up': (90, 180)
        }
    ),
    
    'non_hodgkin_lymphoma': CancerConfig(
        name='Non-Hodgkin Lymphoma',
        category='Hematologic',
        incidence_rate=0.065,
        age_distribution={'mean': 65, 'std': 15},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='Ann_Arbor',
        stages=['I', 'II', 'III', 'IV'],
        stage_probabilities=[0.15, 0.25, 0.30, 0.30],
        treatments={
            'R-CHOP': 0.7,
            'CHOP': 0.3,
            'Rituximab': 0.8,
            'Radiation Therapy': 0.3,
            'Stem Cell Transplant': 0.15
        },
        complications={
            'Neutropenia': 0.5,
            'Infection': 0.3,
            'Tumor Lysis Syndrome': 0.1
        },
        outcomes={
            'Complete Remission': 0.75,
            'Relapse': 0.20,
            'Death': 0.05
        },
        outcome_modifiers={
            'I': {'Complete Remission': 0.90, 'Relapse': 0.08, 'Death': 0.02},
            'IV': {'Complete Remission': 0.60, 'Relapse': 0.30, 'Death': 0.10}
        },
        timeline_days={
            'diagnosis_to_treatment': (7, 28),
            'treatment_cycle': (21, 28),
            'follow_up': (90, 180)
        }
    ),
    
    # THYROID CANCERS
    'papillary_thyroid': CancerConfig(
        name='Papillary Thyroid Carcinoma',
        category='Thyroid',
        incidence_rate=0.065,
        age_distribution={'mean': 45, 'std': 15},
        gender_distribution={'M': 0.25, 'F': 0.75},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N1M1'],
        stage_probabilities=[0.50, 0.25, 0.20, 0.05],
        treatments={
            'Total Thyroidectomy': 0.70,
            'Lobectomy': 0.30,
            'Radioactive Iodine': 0.65,
            'Levothyroxine': 0.95,
            'Central Neck Dissection': 0.25
        },
        complications={
            'Hypoparathyroidism': 0.15,
            'Recurrent Laryngeal Nerve Injury': 0.05,
            'Hypothyroidism': 0.80
        },
        outcomes={
            'Disease Free': 0.92,
            'Recurrence': 0.06,
            'Death': 0.02
        },
        outcome_modifiers={
            'T1N0M0': {'Disease Free': 0.98, 'Recurrence': 0.015, 'Death': 0.005},
            'T4N1M1': {'Disease Free': 0.70, 'Recurrence': 0.25, 'Death': 0.05}
        },
        timeline_days={
            'diagnosis_to_surgery': (14, 45),
            'surgery_to_rai': (30, 90),
            'follow_up': (90, 180)
        }
    ),

    'thyroid_anaplastic': CancerConfig(
        name='Anaplastic Thyroid Carcinoma',
        category='Thyroid',
        incidence_rate=0.002,
        age_distribution={'mean': 70, 'std': 10},
        gender_distribution={'M': 0.45, 'F': 0.55},
        staging_system='TNM',
        stages=['T4aN0M0', 'T4aN1M0', 'T4bN0M0', 'T4bN1M1'],
        stage_probabilities=[0.20, 0.30, 0.30, 0.20],
        treatments={
            'Tracheostomy': 0.60,
            'Palliative Surgery': 0.40,
            'Radiation Therapy': 0.70,
            'Chemotherapy': 0.50,
            'Targeted Therapy': 0.30,
            'Palliative Care': 0.80
        },
        complications={
            'Airway Obstruction': 0.60,
            'Dysphagia': 0.70,
            'Vocal Cord Paralysis': 0.40,
            'Bleeding': 0.30,
            'Pain': 0.80
        },
        outcomes={
            'Stable Disease': 0.20,
            'Progressive Disease': 0.70,
            'Death': 0.10
        },
        outcome_modifiers={
            'T4aN0M0': {'Stable Disease': 0.35, 'Progressive Disease': 0.55, 'Death': 0.10},
            'T4bN1M1': {'Stable Disease': 0.10, 'Progressive Disease': 0.80, 'Death': 0.10}
        },
        timeline_days={
            'diagnosis_to_treatment': (3, 14),
            'treatment_cycle': (21, 28),
            'follow_up': (30, 60)
        }
    ),
    
    # GENITOURINARY CANCERS
    'prostate_cancer': CancerConfig(
        name='Prostate Cancer',
        category='Genitourinary',
        incidence_rate=0.095,
        age_distribution={'mean': 68, 'std': 10},
        gender_distribution={'M': 1.0, 'F': 0.0},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N0M0', 'T4N1M1'],
        stage_probabilities=[0.40, 0.35, 0.20, 0.05],
        treatments={
            'Radical Prostatectomy': 0.40,
            'Radiation Therapy': 0.35,
            'Androgen Deprivation Therapy': 0.25,
            'Active Surveillance': 0.30,
            'Brachytherapy': 0.15,
            'Chemotherapy': 0.10
        },
        complications={
            'Erectile Dysfunction': 0.60,
            'Urinary Incontinence': 0.25,
            'Bowel Dysfunction': 0.15,
            'Hot Flashes': 0.40
        },
        outcomes={
            'No Evidence of Disease': 0.88,
            'Biochemical Recurrence': 0.10,
            'Death': 0.02
        },
        outcome_modifiers={
            'T1N0M0': {'No Evidence of Disease': 0.95, 'Biochemical Recurrence': 0.04, 'Death': 0.01},
            'T4N1M1': {'No Evidence of Disease': 0.40, 'Biochemical Recurrence': 0.35, 'Death': 0.25}
        },
        timeline_days={
            'diagnosis_to_treatment': (30, 90),
            'treatment_duration': (60, 180),
            'follow_up': (90, 180)
        }
    ),
    
    'renal_cell_carcinoma': CancerConfig(
        name='Renal Cell Carcinoma',
        category='Genitourinary',
        incidence_rate=0.035,
        age_distribution={'mean': 64, 'std': 12},
        gender_distribution={'M': 0.65, 'F': 0.35},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N0M0', 'T4N1M1'],
        stage_probabilities=[0.45, 0.25, 0.20, 0.10],
        treatments={
            'Partial Nephrectomy': 0.50,
            'Radical Nephrectomy': 0.40,
            'Targeted Therapy': 0.25,
            'Immunotherapy': 0.20,
            'Ablation': 0.15
        },
        complications={
            'Chronic Kidney Disease': 0.30,
            'Surgical Site Infection': 0.10,
            'Pneumothorax': 0.05
        },
        outcomes={
            'No Evidence of Disease': 0.82,
            'Progressive Disease': 0.15,
            'Death': 0.03
        },
        outcome_modifiers={
            'T1N0M0': {'No Evidence of Disease': 0.95, 'Progressive Disease': 0.04, 'Death': 0.01},
            'T4N1M1': {'No Evidence of Disease': 0.25, 'Progressive Disease': 0.55, 'Death': 0.20}
        },
        timeline_days={
            'diagnosis_to_surgery': (14, 45),
            'recovery': (30, 90),
            'follow_up': (90, 180)
        }
    ),
    
    'bladder_cancer': CancerConfig(
        name='Bladder Cancer',
        category='Genitourinary',
        incidence_rate=0.04,
        age_distribution={'mean': 69, 'std': 11},
        gender_distribution={'M': 0.75, 'F': 0.25},
        staging_system='TNM',
        stages=['Ta', 'T1', 'T2', 'T3', 'T4'],
        stage_probabilities=[0.30, 0.25, 0.20, 0.15, 0.10],
        treatments={
            'TURBT': 0.80,
            'Radical Cystectomy': 0.30,
            'BCG Immunotherapy': 0.40,
            'Chemotherapy': 0.35,
            'Radiation Therapy': 0.20
        },
        complications={
            'UTI': 0.35,
            'Hematuria': 0.25,
            'Urinary Incontinence': 0.40,
            'Bowel Obstruction': 0.10
        },
        outcomes={
            'No Evidence of Disease': 0.70,
            'Recurrence': 0.25,
            'Death': 0.05
        },
        outcome_modifiers={
            'Ta': {'No Evidence of Disease': 0.90, 'Recurrence': 0.08, 'Death': 0.02},
            'T4': {'No Evidence of Disease': 0.35, 'Recurrence': 0.45, 'Death': 0.20}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_cycle': (21, 42),
            'follow_up': (90, 180)
        }
    ),
    
    'testicular_cancer': CancerConfig(
        name='Testicular Cancer',
        category='Genitourinary',
        incidence_rate=0.012,
        age_distribution={'mean': 32, 'std': 8},
        gender_distribution={'M': 1.0, 'F': 0.0},
        staging_system='TNM',
        stages=['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB'],
        stage_probabilities=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
        treatments={
            'Orchiectomy': 0.95,
            'BEP Chemotherapy': 0.60,
            'EP Chemotherapy': 0.30,
            'Radiation Therapy': 0.25,
            'RPLND': 0.20
        },
        complications={
            'Retrograde Ejaculation': 0.15,
            'Neuropathy': 0.20,
            'Pulmonary Toxicity': 0.10,
            'Ototoxicity': 0.15
        },
        outcomes={
            'Complete Remission': 0.95,
            'Relapse': 0.04,
            'Death': 0.01
        },
        outcome_modifiers={
            'IA': {'Complete Remission': 0.99, 'Relapse': 0.008, 'Death': 0.002},
            'IIIB': {'Complete Remission': 0.85, 'Relapse': 0.12, 'Death': 0.03}
        },
        timeline_days={
            'diagnosis_to_surgery': (3, 14),
            'surgery_to_chemo': (14, 28),
            'follow_up': (90, 180)
        }
    ),
    
    # GYNECOLOGIC CANCERS
    'ovarian_cancer': CancerConfig(
        name='Ovarian Cancer',
        category='Gynecologic',
        incidence_rate=0.025,
        age_distribution={'mean': 63, 'std': 13},
        gender_distribution={'M': 0.0, 'F': 1.0},
        staging_system='FIGO',
        stages=['IA', 'IB', 'IC', 'IIA', 'IIIA', 'IIIB', 'IIIC', 'IV'],
        stage_probabilities=[0.10, 0.05, 0.05, 0.05, 0.15, 0.15, 0.35, 0.10],
        treatments={
            'Debulking Surgery': 0.80,
            'Carboplatin + Paclitaxel': 0.75,
            'Bevacizumab': 0.30,
            'PARP Inhibitors': 0.20,
            'Secondary Surgery': 0.25
        },
        complications={
            'Neuropathy': 0.60,
            'Bowel Obstruction': 0.20,
            'Ascites': 0.40,
            'Pleural Effusion': 0.15
        },
        outcomes={
            'Complete Response': 0.45,
            'Partial Response': 0.35,
            'Progressive Disease': 0.20
        },
        outcome_modifiers={
            'IA': {'Complete Response': 0.90, 'Partial Response': 0.08, 'Progressive Disease': 0.02},
            'IIIC': {'Complete Response': 0.30, 'Partial Response': 0.45, 'Progressive Disease': 0.25}
        },
        timeline_days={
            'diagnosis_to_surgery': (7, 21),
            'surgery_to_chemo': (21, 42),
            'chemo_cycle': (21, 21)
        }
    ),
    
    'endometrial_cancer': CancerConfig(
        name='Endometrial Cancer',
        category='Gynecologic',
        incidence_rate=0.048,
        age_distribution={'mean': 62, 'std': 11},
        gender_distribution={'M': 0.0, 'F': 1.0},
        staging_system='FIGO',
        stages=['IA', 'IB', 'II', 'IIIA', 'IIIB', 'IIIC', 'IV'],
        stage_probabilities=[0.50, 0.20, 0.10, 0.08, 0.05, 0.05, 0.02],
        treatments={
            'Hysterectomy': 0.85,
            'Radiation Therapy': 0.40,
            'Chemotherapy': 0.25,
            'Hormone Therapy': 0.30,
            'Lymphadenectomy': 0.60
        },
        complications={
            'Lymphedema': 0.20,
            'Bowel Dysfunction': 0.15,
            'Sexual Dysfunction': 0.30,
            'Vaginal Stenosis': 0.25
        },
        outcomes={
            'Disease Free': 0.85,
            'Recurrence': 0.12,
            'Death': 0.03
        },
        outcome_modifiers={
            'IA': {'Disease Free': 0.95, 'Recurrence': 0.04, 'Death': 0.01},
            'IV': {'Disease Free': 0.40, 'Recurrence': 0.35, 'Death': 0.25}
        },
        timeline_days={
            'diagnosis_to_surgery': (14, 45),
            'surgery_to_adjuvant': (30, 60),
            'follow_up': (90, 180)
        }
    ),

    'cervical_cancer': CancerConfig(
        name='Cervical Cancer',
        category='Gynecologic',
        incidence_rate=0.02,
        age_distribution={'mean': 49, 'std': 15},
        gender_distribution={'M': 0.0, 'F': 1.0},
        staging_system='FIGO',
        stages=['IA1', 'IA2', 'IB1', 'IB2', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IVA', 'IVB'],
        stage_probabilities=[0.08, 0.07, 0.25, 0.15, 0.10, 0.15, 0.08, 0.07, 0.03, 0.02],
        treatments={
            'Radical Hysterectomy': 0.50,
            'Cone Biopsy': 0.20,
            'Pelvic Lymphadenectomy': 0.60,
            'Chemoradiation': 0.55,
            'Cisplatin': 0.65,
            'Brachytherapy': 0.40,
            'Simple Hysterectomy': 0.25
        },
        complications={
            'Lymphedema': 0.25,
            'Bowel Dysfunction': 0.20,
            'Bladder Dysfunction': 0.15,
            'Sexual Dysfunction': 0.40,
            'Renal Toxicity': 0.10,
            'Vaginal Stenosis': 0.30
        },
        outcomes={
            'Disease Free': 0.75,
            'Recurrence': 0.20,
            'Death': 0.05
        },
        outcome_modifiers={
            'IA1': {'Disease Free': 0.98, 'Recurrence': 0.015, 'Death': 0.005},
            'IVB': {'Disease Free': 0.20, 'Recurrence': 0.50, 'Death': 0.30}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 45),
            'surgery_to_adjuvant': (30, 60),
            'chemorad_fraction': (1, 1),
            'follow_up': (90, 180)
        }
    ),
    
    # THORACIC CANCERS
    'lung_adenocarcinoma': CancerConfig(
        name='Lung Adenocarcinoma',
        category='Thoracic',
        incidence_rate=0.065,
        age_distribution={'mean': 66, 'std': 11},
        gender_distribution={'M': 0.52, 'F': 0.48},
        staging_system='TNM',
        stages=['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IV'],
        stage_probabilities=[0.15, 0.10, 0.05, 0.10, 0.15, 0.15, 0.30],
        treatments={
            'Lobectomy': 0.35,
            'Stereotactic Radiosurgery': 0.20,
            'Chemotherapy': 0.60,
            'Targeted Therapy': 0.40,
            'Immunotherapy': 0.30,
            'Radiation Therapy': 0.50
        },
        complications={
            'Pneumonia': 0.25,
            'Atelectasis': 0.20,
            'Air Leak': 0.15,
            'Neuropathy': 0.35
        },
        outcomes={
            'No Evidence of Disease': 0.35,
            'Stable Disease': 0.30,
            'Progressive Disease': 0.35
        },
        outcome_modifiers={
            'IA': {'No Evidence of Disease': 0.80, 'Stable Disease': 0.15, 'Progressive Disease': 0.05},
            'IV': {'No Evidence of Disease': 0.10, 'Stable Disease': 0.30, 'Progressive Disease': 0.60}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_cycle': (21, 28),
            'follow_up': (60, 120)
        }
    ),

    'lung_squamous_cell': CancerConfig(
        name='Lung Squamous Cell Carcinoma',
        category='Thoracic',
        incidence_rate=0.04,
        age_distribution={'mean': 68, 'std': 10},
        gender_distribution={'M': 0.70, 'F': 0.30},
        staging_system='TNM',
        stages=['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC', 'IV'],
        stage_probabilities=[0.12, 0.08, 0.08, 0.12, 0.20, 0.20, 0.10, 0.10],
        treatments={
            'Lobectomy': 0.40,
            'Pneumonectomy': 0.15,
            'Chemotherapy': 0.65,
            'Radiation Therapy': 0.55,
            'Immunotherapy': 0.25,
            'Concurrent Chemoradiation': 0.35
        },
        complications={
            'Pneumonia': 0.30,
            'Respiratory Failure': 0.15,
            'Cardiac Complications': 0.20,
            'Esophagitis': 0.25,
            'Pneumonitis': 0.20
        },
        outcomes={
            'No Evidence of Disease': 0.30,
            'Stable Disease': 0.35,
            'Progressive Disease': 0.35
        },
        outcome_modifiers={
            'IA': {'No Evidence of Disease': 0.75, 'Stable Disease': 0.20, 'Progressive Disease': 0.05},
            'IV': {'No Evidence of Disease': 0.08, 'Stable Disease': 0.32, 'Progressive Disease': 0.60}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_cycle': (21, 28),
            'follow_up': (60, 120)
        }
    ),

    'small_cell_lung_cancer': CancerConfig(
        name='Small Cell Lung Cancer',
        category='Thoracic',
        incidence_rate=0.025,
        age_distribution={'mean': 65, 'std': 10},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='TNM',
        stages=['Limited Stage', 'Extensive Stage'],
        stage_probabilities=[0.30, 0.70],
        treatments={
            'Etoposide + Cisplatin': 0.80,
            'Concurrent Chemoradiation': 0.60,
            'Prophylactic Cranial Irradiation': 0.40,
            'Immunotherapy': 0.20,
            'Palliative Radiation': 0.30
        },
        complications={
            'Neutropenia': 0.70,
            'Thrombocytopenia': 0.50,
            'Pneumonitis': 0.25,
            'Cognitive Impairment': 0.30,
            'Syndrome of Inappropriate ADH': 0.15
        },
        outcomes={
            'Complete Response': 0.25,
            'Partial Response': 0.45,
            'Progressive Disease': 0.30
        },
        outcome_modifiers={
            'Limited Stage': {'Complete Response': 0.40, 'Partial Response': 0.45, 'Progressive Disease': 0.15},
            'Extensive Stage': {'Complete Response': 0.15, 'Partial Response': 0.45, 'Progressive Disease': 0.40}
        },
        timeline_days={
            'diagnosis_to_treatment': (7, 14),
            'treatment_cycle': (21, 28),
            'follow_up': (30, 90)
        }
    ),

    'mesothelioma': CancerConfig(
        name='Pleural Mesothelioma',
        category='Thoracic',
        incidence_rate=0.006,
        age_distribution={'mean': 72, 'std': 10},
        gender_distribution={'M': 0.85, 'F': 0.15},
        staging_system='TNM',
        stages=['IA', 'IB', 'II', 'IIIA', 'IIIB', 'IV'],
        stage_probabilities=[0.10, 0.15, 0.20, 0.25, 0.20, 0.10],
        treatments={
            'Pleurectomy': 0.30,
            'Extrapleural Pneumonectomy': 0.15,
            'Pemetrexed + Cisplatin': 0.60,
            'Radiation Therapy': 0.40,
            'Immunotherapy': 0.25,
            'Pleurodesis': 0.50
        },
        complications={
            'Pleural Effusion': 0.80,
            'Dyspnea': 0.90,
            'Chest Pain': 0.85,
            'Respiratory Failure': 0.25,
            'Cardiac Complications': 0.15
        },
        outcomes={
            'Stable Disease': 0.35,
            'Progressive Disease': 0.60,
            'Death': 0.05
        },
        outcome_modifiers={
            'IA': {'Stable Disease': 0.60, 'Progressive Disease': 0.35, 'Death': 0.05},
            'IV': {'Stable Disease': 0.15, 'Progressive Disease': 0.75, 'Death': 0.10}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_cycle': (21, 28),
            'follow_up': (30, 90)
        }
    ),
    
    # CENTRAL NERVOUS SYSTEM
    'glioblastoma': CancerConfig(
        name='Glioblastoma',
        category='Central Nervous System',
        incidence_rate=0.012,
        age_distribution={'mean': 64, 'std': 12},
        gender_distribution={'M': 0.60, 'F': 0.40},
        staging_system='WHO',
        stages=['Grade IV'],
        stage_probabilities=[1.0],
        treatments={
            'Surgical Resection': 0.80,
            'Radiation Therapy': 0.90,
            'Temozolomide': 0.85,
            'Bevacizumab': 0.30,
            'Tumor Treating Fields': 0.20
        },
        complications={
            'Seizures': 0.40,
            'Cognitive Impairment': 0.60,
            'Hemiparesis': 0.30,
            'Aphasia': 0.25
        },
        outcomes={
            'Stable Disease': 0.35,
            'Progressive Disease': 0.60,
            'Death': 0.05
        },
        outcome_modifiers={
            'Grade IV': {'Stable Disease': 0.35, 'Progressive Disease': 0.60, 'Death': 0.05}
        },
        timeline_days={
            'diagnosis_to_surgery': (3, 14),
            'surgery_to_radiation': (14, 28),
            'follow_up': (30, 60)
        }
    ),

    'meningioma': CancerConfig(
        name='Meningioma',
        category='Central Nervous System',
        incidence_rate=0.015,
        age_distribution={'mean': 58, 'std': 15},
        gender_distribution={'M': 0.35, 'F': 0.65},
        staging_system='WHO',
        stages=['Grade I', 'Grade II', 'Grade III'],
        stage_probabilities=[0.80, 0.17, 0.03],
        treatments={
            'Observation': 0.40,
            'Surgical Resection': 0.55,
            'Stereotactic Radiosurgery': 0.25,
            'Radiation Therapy': 0.20,
            'Chemotherapy': 0.05
        },
        complications={
            'Seizures': 0.15,
            'Neurological Deficit': 0.20,
            'CSF Leak': 0.05,
            'Infection': 0.05,
            'Cognitive Impairment': 0.10
        },
        outcomes={
            'Stable Disease': 0.85,
            'Recurrence': 0.12,
            'Death': 0.03
        },
        outcome_modifiers={
            'Grade I': {'Stable Disease': 0.95, 'Recurrence': 0.04, 'Death': 0.01},
            'Grade III': {'Stable Disease': 0.60, 'Recurrence': 0.35, 'Death': 0.05}
        },
        timeline_days={
            'diagnosis_to_treatment': (30, 90),
            'follow_up': (180, 365)
        }
    ),

    'pituitary_adenoma': CancerConfig(
        name='Pituitary Adenoma',
        category='Central Nervous System',
        incidence_rate=0.012,
        age_distribution={'mean': 45, 'std': 15},
        gender_distribution={'M': 0.45, 'F': 0.55},
        staging_system='Size',
        stages=['Microadenoma', 'Macroadenoma'],
        stage_probabilities=[0.60, 0.40],
        treatments={
            'Transsphenoidal Surgery': 0.60,
            'Medical Therapy': 0.40,
            'Radiation Therapy': 0.20,
            'Observation': 0.30,
            'Hormone Replacement': 0.50
        },
        complications={
            'Hypopituitarism': 0.25,
            'Diabetes Insipidus': 0.15,
            'Visual Field Defects': 0.30,
            'CSF Leak': 0.05,
            'Rhinorrhea': 0.10
        },
        outcomes={
            'Remission': 0.75,
            'Stable Disease': 0.20,
            'Recurrence': 0.05
        },
        outcome_modifiers={
            'Microadenoma': {'Remission': 0.85, 'Stable Disease': 0.13, 'Recurrence': 0.02},
            'Macroadenoma': {'Remission': 0.60, 'Stable Disease': 0.30, 'Recurrence': 0.10}
        },
        timeline_days={
            'diagnosis_to_treatment': (30, 90),
            'follow_up': (90, 180)
        }
    ),
    
    # GASTROINTESTINAL CANCERS
    'colorectal_cancer': CancerConfig(
        name='Colorectal Cancer',
        category='Gastrointestinal',
        incidence_rate=0.12,
        age_distribution={'mean': 68, 'std': 12},
        gender_distribution={'M': 0.52, 'F': 0.48},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'],
        stage_probabilities=[0.20, 0.30, 0.35, 0.15],
        treatments={
            'Surgery': 0.80,
            'FOLFOX': 0.45,
            'FOLFIRI': 0.35,
            'Bevacizumab': 0.25,
            'Radiation Therapy': 0.30,
            'Cetuximab': 0.20
        },
        complications={
            'Anastomotic Leak': 0.15,
            'Surgical Site Infection': 0.20,
            'Neuropathy': 0.40,
            'Diarrhea': 0.35
        },
        outcomes={
            'No Evidence of Disease': 0.65,
            'Recurrence': 0.25,
            'Death': 0.10
        },
        outcome_modifiers={
            'T1N0M0': {'No Evidence of Disease': 0.92, 'Recurrence': 0.06, 'Death': 0.02},
            'T4N2M1': {'No Evidence of Disease': 0.25, 'Recurrence': 0.45, 'Death': 0.30}
        },
        timeline_days={
            'diagnosis_to_surgery': (14, 45),
            'surgery_to_chemo': (30, 60),
            'chemo_cycle': (14, 21)
        }
    ),
    
    'pancreatic_cancer': CancerConfig(
        name='Pancreatic Adenocarcinoma',
        category='Gastrointestinal',
        incidence_rate=0.032,
        age_distribution={'mean': 70, 'std': 10},
        gender_distribution={'M': 0.52, 'F': 0.48},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N1M1'],
        stage_probabilities=[0.05, 0.10, 0.35, 0.50],
        treatments={
            'Whipple Procedure': 0.20,
            'FOLFIRINOX': 0.35,
            'Gemcitabine + nab-Paclitaxel': 0.40,
            'Palliative Care': 0.40,
            'Radiation Therapy': 0.25
        },
        complications={
            'Pancreatic Fistula': 0.25,
            'Delayed Gastric Emptying': 0.20,
            'Diabetes': 0.60,
            'Malabsorption': 0.40
        },
        outcomes={
            'Stable Disease': 0.25,
            'Progressive Disease': 0.65,
            'Death': 0.10
        },
        outcome_modifiers={
            'T1N0M0': {'Stable Disease': 0.60, 'Progressive Disease': 0.30, 'Death': 0.10},
            'T4N1M1': {'Stable Disease': 0.10, 'Progressive Disease': 0.75, 'Death': 0.15}
        },
        timeline_days={
            'diagnosis_to_treatment': (7, 21),
            'treatment_cycle': (14, 21),
            'follow_up': (30, 60)
        }
    ),

    'gastric_cancer': CancerConfig(
        name='Gastric Cancer',
        category='Gastrointestinal',
        incidence_rate=0.032,
        age_distribution={'mean': 68, 'std': 12},
        gender_distribution={'M': 0.65, 'F': 0.35},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'],
        stage_probabilities=[0.15, 0.20, 0.35, 0.30],
        treatments={
            'Gastrectomy': 0.70,
            'Chemotherapy': 0.60,
            'Radiation Therapy': 0.25,
            'Targeted Therapy': 0.20,
            'Endoscopic Resection': 0.15
        },
        complications={
            'Anastomotic Leak': 0.20,
            'Dumping Syndrome': 0.40,
            'Malnutrition': 0.50,
            'Infection': 0.15,
            'B12 Deficiency': 0.60
        },
        outcomes={
            'Disease Free': 0.45,
            'Recurrence': 0.40,
            'Death': 0.15
        },
        outcome_modifiers={
            'T1N0M0': {'Disease Free': 0.85, 'Recurrence': 0.12, 'Death': 0.03},
            'T4N2M1': {'Disease Free': 0.15, 'Recurrence': 0.55, 'Death': 0.30}
        },
        timeline_days={
            'diagnosis_to_surgery': (14, 30),
            'surgery_to_chemo': (30, 60),
            'follow_up': (90, 180)
        }
    ),

    'esophageal_cancer': CancerConfig(
        name='Esophageal Cancer',
        category='Gastrointestinal',
        incidence_rate=0.02,
        age_distribution={'mean': 67, 'std': 11},
        gender_distribution={'M': 0.80, 'F': 0.20},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'],
        stage_probabilities=[0.10, 0.15, 0.40, 0.35],
        treatments={
            'Esophagectomy': 0.40,
            'Chemoradiation': 0.60,
            'Neoadjuvant Chemotherapy': 0.50,
            'Endoscopic Resection': 0.10,
            'Palliative Stenting': 0.25
        },
        complications={
            'Anastomotic Leak': 0.25,
            'Pneumonia': 0.30,
            'Cardiac Complications': 0.20,
            'Dysphagia': 0.60,
            'Aspiration': 0.20
        },
        outcomes={
            'Disease Free': 0.30,
            'Recurrence': 0.50,
            'Death': 0.20
        },
        outcome_modifiers={
            'T1N0M0': {'Disease Free': 0.80, 'Recurrence': 0.15, 'Death': 0.05},
            'T4N2M1': {'Disease Free': 0.10, 'Recurrence': 0.60, 'Death': 0.30}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'neoadjuvant_to_surgery': (60, 90),
            'follow_up': (60, 120)
        }
    ),

    'hepatocellular_carcinoma': CancerConfig(
        name='Hepatocellular Carcinoma',
        category='Gastrointestinal',
        incidence_rate=0.028,
        age_distribution={'mean': 65, 'std': 12},
        gender_distribution={'M': 0.75, 'F': 0.25},
        staging_system='Barcelona Clinic',
        stages=['Very Early (0)', 'Early (A)', 'Intermediate (B)', 'Advanced (C)', 'Terminal (D)'],
        stage_probabilities=[0.10, 0.25, 0.30, 0.25, 0.10],
        treatments={
            'Surgical Resection': 0.25,
            'Liver Transplantation': 0.15,
            'Radiofrequency Ablation': 0.30,
            'TACE': 0.40,
            'Sorafenib': 0.25,
            'Immunotherapy': 0.20
        },
        complications={
            'Liver Failure': 0.20,
            'Portal Hypertension': 0.40,
            'Ascites': 0.50,
            'Variceal Bleeding': 0.15,
            'Hepatic Encephalopathy': 0.25
        },
        outcomes={
            'Stable Disease': 0.40,
            'Progressive Disease': 0.50,
            'Death': 0.10
        },
        outcome_modifiers={
            'Very Early (0)': {'Stable Disease': 0.80, 'Progressive Disease': 0.15, 'Death': 0.05},
            'Advanced (C)': {'Stable Disease': 0.15, 'Progressive Disease': 0.70, 'Death': 0.15}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_cycle': (28, 42),
            'follow_up': (60, 120)
        }
    ),

    'cholangiocarcinoma': CancerConfig(
        name='Cholangiocarcinoma',
        category='Gastrointestinal',
        incidence_rate=0.012,
        age_distribution={'mean': 70, 'std': 10},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'],
        stage_probabilities=[0.10, 0.15, 0.35, 0.40],
        treatments={
            'Surgical Resection': 0.30,
            'Liver Transplantation': 0.10,
            'Chemotherapy': 0.60,
            'Radiation Therapy': 0.25,
            'Biliary Stenting': 0.50,
            'Targeted Therapy': 0.15
        },
        complications={
            'Biliary Obstruction': 0.70,
            'Cholangitis': 0.30,
            'Liver Failure': 0.15,
            'Malnutrition': 0.40,
            'Pruritus': 0.60
        },
        outcomes={
            'Stable Disease': 0.25,
            'Progressive Disease': 0.65,
            'Death': 0.10
        },
        outcome_modifiers={
            'T1N0M0': {'Stable Disease': 0.60, 'Progressive Disease': 0.30, 'Death': 0.10},
            'T4N2M1': {'Stable Disease': 0.10, 'Progressive Disease': 0.75, 'Death': 0.15}
        },
        timeline_days={
            'diagnosis_to_treatment': (7, 21),
            'treatment_cycle': (21, 28),
            'follow_up': (30, 90)
        }
    ),
    
    # DERMATOLOGIC CANCERS
    'melanoma': CancerConfig(
        name='Melanoma',
        category='Dermatologic',
        incidence_rate=0.024,
        age_distribution={'mean': 58, 'std': 16},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='TNM',
        stages=['IA', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC', 'IV'],
        stage_probabilities=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02],
        treatments={
            'Wide Local Excision': 0.85,
            'Sentinel Lymph Node Biopsy': 0.60,
            'Interferon': 0.25,
            'Immunotherapy': 0.30,
            'Targeted Therapy': 0.20,
            'Radiation Therapy': 0.15
        },
        complications={
            'Wound Infection': 0.15,
            'Lymphedema': 0.20,
            'Autoimmune Reactions': 0.25,
            'Vitiligo': 0.10,
            'Fatigue': 0.50
        },
        outcomes={
            'No Evidence of Disease': 0.75,
            'Recurrence': 0.20,
            'Death': 0.05
        },
        outcome_modifiers={
            'IA': {'No Evidence of Disease': 0.95, 'Recurrence': 0.04, 'Death': 0.01},
            'IV': {'No Evidence of Disease': 0.25, 'Recurrence': 0.45, 'Death': 0.30}
        },
        timeline_days={
            'diagnosis_to_surgery': (7, 21),
            'surgery_to_adjuvant': (30, 60),
            'follow_up': (90, 180)
        }
    ),

    'basal_cell_carcinoma': CancerConfig(
        name='Basal Cell Carcinoma',
        category='Dermatologic',
        incidence_rate=0.016,
        age_distribution={'mean': 65, 'std': 12},
        gender_distribution={'M': 0.55, 'F': 0.45},
        staging_system='TNM',
        stages=['T1', 'T2', 'T3', 'T4'],
        stage_probabilities=[0.70, 0.20, 0.08, 0.02],
        treatments={
            'Excision': 0.70,
            'Mohs Surgery': 0.40,
            'Curettage and Electrodesiccation': 0.20,
            'Cryotherapy': 0.15,
            'Topical Therapy': 0.10,
            'Radiation Therapy': 0.05
        },
        complications={
            'Wound Infection': 0.05,
            'Scarring': 0.30,
            'Nerve Damage': 0.02,
            'Recurrence': 0.05
        },
        outcomes={
            'Cure': 0.95,
            'Local Recurrence': 0.04,
            'Metastasis': 0.01
        },
        outcome_modifiers={
            'T1': {'Cure': 0.98, 'Local Recurrence': 0.019, 'Metastasis': 0.001},
            'T4': {'Cure': 0.85, 'Local Recurrence': 0.12, 'Metastasis': 0.03}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 45),
            'follow_up': (90, 365)
        }
    ),
    
    # HEAD AND NECK CANCERS
    'head_neck_squamous': CancerConfig(
        name='Head and Neck Squamous Cell Carcinoma',
        category='Head and Neck',
        incidence_rate=0.02,
        age_distribution={'mean': 62, 'std': 11},
        gender_distribution={'M': 0.75, 'F': 0.25},
        staging_system='TNM',
        stages=['T1N0M0', 'T2N0M0', 'T3N1M0', 'T4N2M1'],
        stage_probabilities=[0.20, 0.25, 0.35, 0.20],
        treatments={
            'Surgery': 0.60,
            'Radiation Therapy': 0.70,
            'Chemotherapy': 0.50,
            'Immunotherapy': 0.25,
            'Concurrent Chemoradiation': 0.45,
            'Neck Dissection': 0.40
        },
        complications={
            'Xerostomia': 0.70,
            'Dysphagia': 0.50,
            'Mucositis': 0.60,
            'Osteoradionecrosis': 0.10,
            'Lymphedema': 0.25,
            'Trismus': 0.30
        },
        outcomes={
            'Disease Free': 0.60,
            'Recurrence': 0.30,
            'Death': 0.10
        },
        outcome_modifiers={
            'T1N0M0': {'Disease Free': 0.85, 'Recurrence': 0.12, 'Death': 0.03},
            'T4N2M1': {'Disease Free': 0.25, 'Recurrence': 0.50, 'Death': 0.25}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_duration': (42, 70),
            'follow_up': (60, 120)
        }
    ),
    
    # MUSCULOSKELETAL CANCERS
    'osteosarcoma': CancerConfig(
        name='Osteosarcoma',
        category='Musculoskeletal',
        incidence_rate=0.004,
        age_distribution={'mean': 16, 'std': 8},
        gender_distribution={'M': 0.56, 'F': 0.44},
        staging_system='Enneking',
        stages=['IA', 'IB', 'IIA', 'IIB', 'III'],
        stage_probabilities=[0.10, 0.15, 0.25, 0.35, 0.15],
        treatments={
            'Neoadjuvant Chemotherapy': 0.90,
            'Limb Salvage Surgery': 0.75,
            'Amputation': 0.25,
            'Adjuvant Chemotherapy': 0.85,
            'Methotrexate': 0.80,
            'Doxorubicin': 0.80,
            'Cisplatin': 0.70
        },
        complications={
            'Infection': 0.20,
            'Phantom Pain': 0.40,
            'Prosthetic Complications': 0.30,
            'Pulmonary Toxicity': 0.15,
            'Cardiac Toxicity': 0.10,
            'Renal Toxicity': 0.15
        },
        outcomes={
            'Disease Free': 0.70,
            'Recurrence': 0.25,
            'Death': 0.05
        },
        outcome_modifiers={
            'IA': {'Disease Free': 0.90, 'Recurrence': 0.08, 'Death': 0.02},
            'III': {'Disease Free': 0.40, 'Recurrence': 0.45, 'Death': 0.15}
        },
        timeline_days={
            'diagnosis_to_chemo': (7, 14),
            'neoadjuvant_duration': (70, 90),
            'chemo_to_surgery': (14, 21),
            'surgery_to_adjuvant': (21, 35),
            'follow_up': (90, 180)
        }
    ),

    'soft_tissue_sarcoma': CancerConfig(
        name='Soft Tissue Sarcoma',
        category='Musculoskeletal',
        incidence_rate=0.008,
        age_distribution={'mean': 58, 'std': 18},
        gender_distribution={'M': 0.52, 'F': 0.48},
        staging_system='TNM',
        stages=['T1aN0M0', 'T1bN0M0', 'T2aN0M0', 'T2bN0M0', 'T2bN1M1'],
        stage_probabilities=[0.20, 0.25, 0.25, 0.20, 0.10],
        treatments={
            'Wide Local Excision': 0.80,
            'Radiation Therapy': 0.50,
            'Chemotherapy': 0.40,
            'Amputation': 0.10,
            'Targeted Therapy': 0.20
        },
        complications={
            'Wound Complications': 0.25,
            'Lymphedema': 0.15,
            'Fibrosis': 0.30,
            'Functional Impairment': 0.40,
            'Neuropathy': 0.20
        },
        outcomes={
            'Disease Free': 0.65,
            'Recurrence': 0.25,
            'Death': 0.10
        },
        outcome_modifiers={
            'T1aN0M0': {'Disease Free': 0.90, 'Recurrence': 0.08, 'Death': 0.02},
            'T2bN1M1': {'Disease Free': 0.25, 'Recurrence': 0.50, 'Death': 0.25}
        },
        timeline_days={
            'diagnosis_to_treatment': (14, 30),
            'treatment_duration': (30, 60),
            'follow_up': (90, 180)
        }
    )
}

# =================================================================
# 3. MODULAR PATHWAY GENERATORS
# =================================================================

class CancerPathwayGenerator:
    """Base class for cancer-specific pathway generation."""
    
    def __init__(self, config: CancerConfig):
        self.config = config
    
    def generate_patient_demographics(self) -> Dict:
        """Generate patient demographics based on cancer config."""
        age = np.random.normal(
            self.config.age_distribution['mean'],
            self.config.age_distribution['std']
        )
        age = max(18, min(90, int(age)))
        
        gender = np.random.choice(
            ['M', 'F'],
            p=[self.config.gender_distribution['M'], self.config.gender_distribution['F']]
        )
        
        stage = np.random.choice(
            self.config.stages,
            p=self.config.stage_probabilities
        )
        
        return {
            'age': age,
            'gender': gender,
            'stage': stage,
            'cancer_type': self.config.name,
            'cancer_category': self.config.category
        }
    
    def generate_pathway_events(self, patient: Dict, patient_id: str, start_time: datetime) -> List[Dict]:
        """Generate cancer-specific clinical pathway."""
        events = []
        event_id = 0
        current_time = start_time
        
        # Diagnosis event
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnosis',
            'event_subtype': self.config.name,
            'timestamp': current_time,
            'days_from_start': 0
        })
        event_id += 1
        
        # Generate cancer-specific events
        events, event_id, current_time = self._generate_specific_pathway(
            events, event_id, patient, patient_id, current_time, start_time
        )
        
        return events
    
    def _generate_specific_pathway(self, events: List[Dict], event_id: int, 
                                 patient: Dict, patient_id: str, 
                                 current_time: datetime, start_time: datetime) -> Tuple[List[Dict], int, datetime]:
        """Generate pathway events - can be overridden for specific cancers."""
        
        # Staging/diagnostic workup
        current_time += timedelta(days=np.random.randint(7, 21))
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'diagnostic',
            'event_subtype': f'{self.config.staging_system} Staging',
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        
        # Treatments
        for treatment, probability in self.config.treatments.items():
            if np.random.random() < probability:
                treatment_delay = np.random.randint(*self.config.timeline_days.get('diagnosis_to_treatment', (14, 45)))
                current_time += timedelta(days=treatment_delay)
                
                treatment_type = 'surgery' if any(surg in treatment.lower() for surg in ['surgery', 'ectomy', 'resection']) else 'treatment'
                
                events.append({
                    'event_id': event_id,
                    'subject_id': patient_id,
                    'event_type': treatment_type,
                    'event_subtype': treatment,
                    'timestamp': current_time,
                    'days_from_start': (current_time - start_time).days
                })
                event_id += 1
        
        # Complications
        for complication, probability in self.config.complications.items():
            if np.random.random() < probability:
                current_time += timedelta(days=np.random.randint(7, 45))
                events.append({
                    'event_id': event_id,
                    'subject_id': patient_id,
                    'event_type': 'complication',
                    'event_subtype': complication,
                    'timestamp': current_time,
                    'days_from_start': (current_time - start_time).days
                })
                event_id += 1
        
        # Outcome
        stage = patient['stage']
        if stage in self.config.outcome_modifiers:
            outcome_probs = self.config.outcome_modifiers[stage]
            outcomes = list(outcome_probs.keys())
            probabilities = list(outcome_probs.values())
        else:
            outcomes = list(self.config.outcomes.keys())
            probabilities = list(self.config.outcomes.values())
        
        outcome = np.random.choice(outcomes, p=probabilities)
        
        follow_up_delay = np.random.randint(*self.config.timeline_days.get('follow_up', (90, 180)))
        current_time += timedelta(days=follow_up_delay)
        
        events.append({
            'event_id': event_id,
            'subject_id': patient_id,
            'event_type': 'outcome',
            'event_subtype': outcome,
            'timestamp': current_time,
            'days_from_start': (current_time - start_time).days
        })
        event_id += 1
        
        return events, event_id, current_time

# =================================================================
# 4. CANCER REGISTRY MANAGER
# =================================================================

class CancerRegistryManager:
    """Manages the cancer registry and pathway generators."""
    
    def __init__(self):
        self.registry = CANCER_REGISTRY
        self.pathway_generators = self._initialize_generators()
    
    def _initialize_generators(self) -> Dict[str, CancerPathwayGenerator]:
        """Initialize pathway generators for all cancer types."""
        generators = {}
        
        for cancer_key, config in self.registry.items():
            generators[cancer_key] = CancerPathwayGenerator(config)
        
        return generators
    
    def get_cancer_types(self) -> List[str]:
        """Get list of all available cancer types."""
        return [config.name for config in self.registry.values()]
    
    def get_cancer_categories(self) -> List[str]:
        """Get list of all cancer categories."""
        return list(set(config.category for config in self.registry.values()))
    
    def generate_patient_cohort(self, n_patients: int = 500) -> Tuple[List[Dict], List[Dict]]:
        """Generate a cohort of patients with various cancer types."""
        
        # Calculate number of patients per cancer type
        cancer_keys = list(self.registry.keys())
        cancer_rates = [config.incidence_rate for config in self.registry.values()]
        
        # Normalize rates to ensure they sum to 1.0
        total_rate = sum(cancer_rates)
        cancer_probs = [rate / total_rate for rate in cancer_rates]
        
        # Generate patients
        patients = []
        all_events = []
        
        for i in range(n_patients):
            # Select cancer type
            cancer_key = np.random.choice(cancer_keys, p=cancer_probs)
            generator = self.pathway_generators[cancer_key]
            
            # Generate patient demographics
            patient = generator.generate_patient_demographics()
            patient['subject_id'] = f'P{i:04d}'
            patient['first_admission'] = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095))
            
            patients.append(patient)
            
            # Generate pathway events
            events = generator.generate_pathway_events(
                patient, patient['subject_id'], patient['first_admission']
            )
            all_events.extend(events)
        
        return patients, all_events

# Verification function
def verify_registry():
    """Verify that the cancer registry rates sum to approximately 1.0."""
    total_rate = sum(config.incidence_rate for config in CANCER_REGISTRY.values())
    print(f"Total incidence rate: {total_rate:.3f}")
    print(f"Number of cancer types: {len(CANCER_REGISTRY)}")
    print(f"Cancer categories: {len(set(config.category for config in CANCER_REGISTRY.values()))}")
    
    if abs(total_rate - 1.0) > 0.05:
        print("⚠️  Warning: Incidence rates don't sum to 1.0")
    else:
        print("✅ Incidence rates are properly normalized")

if __name__ == "__main__":
    verify_registry()