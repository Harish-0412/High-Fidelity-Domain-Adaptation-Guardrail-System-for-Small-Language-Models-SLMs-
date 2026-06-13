"""
Download medical prescription datasets for the RAG system.

This script downloads the following datasets:
1. DailyMed SPL Drug Labels (NLM)
2. India Medicines & Drug Info Dataset (Kaggle)
3. English Prescribing Dataset (EPD) - NHS England
4. Drugs@FDA Database
5. GP Prescribing Data (Northern Ireland)
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL to destination."""
    try:
        logger.info(f"Downloading {url} to {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)

        print()  # New line after progress
        logger.info(f"Successfully downloaded to {destination}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file."""
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info(f"Successfully extracted to {extract_to}")
        return True

    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_dailymed(output_dir: Path) -> bool:
    """
    Download DailyMed SPL Drug Labels.
    
    Note: DailyMed provides a large dataset (150,000+ drug labels).
    For this implementation, we'll download a sample or provide instructions.
    """
    logger.info("Setting up DailyMed dataset download...")

    # DailyMed provides bulk downloads via FTP
    # For now, we'll create a placeholder with instructions
    dailymed_dir = output_dir / "dailymed"
    dailymed_dir.mkdir(parents=True, exist_ok=True)

    readme = """DailyMed SPL Drug Labels Dataset
=====================================

Source: National Library of Medicine (NLM)
URL: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

This dataset contains 150,000+ FDA-approved drug labels with full prescribing information.

To download the complete dataset:
1. Visit: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
2. Download the SPL XML files
3. Extract to this directory

For initial testing, you can download a smaller sample from:
https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm

Contents:
- Medication Guides
- Prescribing Information
- Dosage & Administration
- Adverse Reactions
- Drug Interactions
- Warnings & Precautions
"""

    (dailymed_dir / "README.txt").write_text(readme)
    logger.info("Created DailyMed directory with instructions")
    return True


def download_india_medicines(output_dir: Path) -> bool:
    """
    Download India Medicines & Drug Info Dataset from Kaggle.
    
    Requires Kaggle API credentials.
    """
    logger.info("Setting up India Medicines dataset...")

    india_dir = output_dir / "india_medicines"
    india_dir.mkdir(parents=True, exist_ok=True)

    readme = """India Medicines & Drug Info Dataset
======================================

Source: Kaggle (scraped from Tata 1mg)
URL: https://www.kaggle.com/datasets/apkaayush/india-medicines-and-drug-info-dataset

This dataset contains ~300,000 medicines with detailed information.

To download:
1. Install Kaggle API: pip install kaggle
2. Setup Kaggle credentials: kaggle configure
3. Download: kaggle datasets download -d apkaayush/india-medicines-and-drug-info-dataset
4. Extract to this directory

Contents:
- Medicine names
- Generic names
- Brand names
- Prices (INR)
- Compositions/ingredients
- Pharmaceutical categories

License: CC BY-NC-SA 4.0
"""

    (india_dir / "README.txt").write_text(readme)
    logger.info("Created India Medicines directory with instructions")
    return True


def download_epd_nhs(output_dir: Path) -> bool:
    """
    Download English Prescribing Dataset (EPD) from NHS England.
    """
    logger.info("Setting up NHS English Prescribing Dataset...")

    epd_dir = output_dir / "nhs_epd"
    epd_dir.mkdir(parents=True, exist_ok=True)

    readme = """English Prescribing Dataset (EPD)
===================================

Source: NHS Business Services Authority (NHSBSA)
URL: https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code

This dataset contains 17M+ rows/month of community prescription data.

To download:
1. Visit: https://opendata.nhsbsa.net/dataset/english-prescribing-dataset-epd-with-snomed-code
2. Register for an account (free)
3. Download the monthly CSV files
4. Extract to this directory

Contents:
- Prescriptions by GP practice
- BNF (British National Formulary) codes
- SNOMED CT codes
- Costs and quantities
- Regional prescribing trends
"""

    (epd_dir / "README.txt").write_text(readme)
    logger.info("Created NHS EPD directory with instructions")
    return True


def download_drugs_fda(output_dir: Path) -> bool:
    """
    Download Drugs@FDA Database.
    """
    logger.info("Setting up Drugs@FDA Database...")

    fda_dir = output_dir / "drugs_fda"
    fda_dir.mkdir(parents=True, exist_ok=True)

    # Download the Drugs@FDA dataset from data.gov
    url = "https://download.open.fda.gov/drug/labeling/drugs@fda-dataset.zip"
    zip_path = output_dir / "drugs_fda_dataset.zip"

    if download_file(url, zip_path):
        if extract_zip(zip_path, fda_dir):
            # Clean up zip file
            zip_path.unlink()
            logger.info("Successfully downloaded and extracted Drugs@FDA dataset")
            return True

    # Fallback to instructions if download fails
    readme = """Drugs@FDA Database
==================

Source: U.S. FDA
URL: https://catalog.data.gov/dataset/drugsfda-database

This dataset contains all FDA-approved drugs since 1939.

To download manually:
1. Visit: https://catalog.data.gov/dataset/drugsfda-database
2. Download the dataset (ZIP file)
3. Extract to this directory

Contents:
- Brand and generic drug info
- Approval history
- Labels and patient information
- Reviews and approval letters
"""

    (fda_dir / "README.txt").write_text(readme)
    logger.info("Created Drugs@FDA directory with instructions")
    return True


def download_gp_ni(output_dir: Path) -> bool:
    """
    Download GP Prescribing Data from Northern Ireland.
    """
    logger.info("Setting up GP Prescribing Data (Northern Ireland)...")

    gp_dir = output_dir / "gp_prescribing_ni"
    gp_dir.mkdir(parents=True, exist_ok=True)

    readme = """GP Prescribing Data (Northern Ireland)
==========================================

Source: Health and Social Care Northern Ireland
URL: https://admin.opendatani.gov.uk/dataset/gp-prescribing-data

This dataset contains monthly GP prescribing data.

To download:
1. Visit: https://admin.opendatani.gov.uk/dataset/gp-prescribing-data
2. Download the monthly CSV files
3. Extract to this directory

Contents:
- Medicine names
- Quantities
- Costs
- By practice
"""

    (gp_dir / "README.txt").write_text(readme)
    logger.info("Created GP Prescribing NI directory with instructions")
    return True


def download_sample_drug_data(output_dir: Path) -> bool:
    """
    Download sample drug data for immediate testing.
    This provides a small dataset to test the pipeline while large datasets are being downloaded.
    """
    logger.info("Downloading sample drug data for testing...")

    sample_dir = output_dir / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create sample drug information files for testing
    sample_drugs = {
        "aspirin.txt": """
ASPIRIN (acetylsalicylic acid)

INDICATIONS AND USAGE:
Aspirin is indicated for:
- Relief of mild to moderate pain (headache, toothache, muscle aches)
- Reduction of fever
- Anti-inflammatory treatment in rheumatoid arthritis and osteoarthritis
- Cardiovascular protection (low-dose aspirin for prevention of myocardial infarction and stroke)

DOSAGE AND ADMINISTRATION:
Adults: 325-650 mg every 4-6 hours as needed for pain/fever
Cardiovascular protection: 81-325 mg daily
Maximum dose: 4 g in 24 hours

CONTRAINDICATIONS:
- Hypersensitivity to aspirin or other NSAIDs
- Active peptic ulcer disease
- Bleeding disorders
- Children and teenagers with viral infections (Reye's syndrome risk)

ADVERSE REACTIONS:
Common: Gastrointestinal upset, nausea, dyspepsia
Serious: GI bleeding, ulceration, bronchospasm, anaphylaxis

DRUG INTERACTIONS:
- Anticoagulants (increased bleeding risk)
- Other NSAIDs (increased GI toxicity)
- Corticosteroids (increased ulcer risk)
- ACE inhibitors (reduced antihypertensive effect)
""",
        "metformin.txt": """
METFORMIN hydrochloride

INDICATIONS AND USAGE:
Metformin is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.

DOSAGE AND ADMINISTRATION:
Starting dose: 500 mg twice daily or 850 mg once daily
Maintenance dose: 2000 mg daily (max 2550 mg)
Take with meals to reduce gastrointestinal side effects

CONTRAINDICATIONS:
- Renal impairment (eGFR < 30 mL/min/1.73 m²)
- Acute or chronic metabolic acidosis
- Hypersensitivity to metformin

ADVERSE REACTIONS:
Common: Diarrhea, nausea, vomiting, abdominal discomfort, metallic taste
Serious: Lactic acidosis (rare but serious), vitamin B12 deficiency

DRUG INTERACTIONS:
- Cimetidine (increased metformin levels)
- Furosemide (increased metformin levels)
- Nifedipine (increased metformin absorption)
- Alcohol (increased lactic acidosis risk)
""",
        "lisinopril.txt": """
LISINOPRIL

INDICATIONS AND USAGE:
Lisinopril is indicated for:
- Treatment of hypertension
- Management of heart failure
- Acute myocardial infarction (within 24 hours)

DOSAGE AND ADMINISTRATION:
Hypertension: Initial 10 mg once daily, maintenance 20-40 mg once daily
Heart failure: Initial 5 mg once daily, maintenance 5-40 mg once daily
Post-MI: Initial 5 mg within 24 hours, then 5 mg at 24 hours, 10 mg at 48 hours, then 10 mg daily

CONTRAINDICATIONS:
- History of angioedema related to ACE inhibitor therapy
- Hereditary or idiopathic angioedema
- Pregnancy (second and third trimesters)

ADVERSE REACTIONS:
Common: Dry cough, headache, dizziness, hypotension
Serious: Angioedema, hyperkalemia, renal impairment

DRUG INTERACTIONS:
- Potassium supplements (hyperkalemia risk)
- Diuretics (increased hypotension risk)
- Lithium (increased lithium levels)
- NSAIDs (reduced antihypertensive effect)
"""
    }

    for filename, content in sample_drugs.items():
        (sample_dir / filename).write_text(content)

    logger.info(f"Created sample drug data in {sample_dir}")
    return True


def main():
    """Main function to download all datasets."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw" / "medical_prescription"

    logger.info(f"Downloading medical prescription datasets to {output_dir}")

    # Download sample data first for immediate testing
    download_sample_drug_data(output_dir)

    # Setup instructions for large datasets
    download_dailymed(output_dir)
    download_india_medicines(output_dir)
    download_epd_nhs(output_dir)
    download_drugs_fda(output_dir)
    download_gp_ni(output_dir)

    logger.info("Dataset download setup complete")
    logger.info("\nNOTE: Large datasets require manual download due to size and authentication requirements.")
    logger.info("Sample data has been provided for immediate testing.")
    logger.info(f"Please check the README.txt files in each subdirectory under {output_dir}")


if __name__ == "__main__":
    main()
