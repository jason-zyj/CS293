# CS293
# Installation
## Clone the repo
'''
git clone https://github.com/jason-zyj/CS293.git
cd CS293
'''

## Make & activate a virtual environment
'''
python3.8 -m venv edu
source edu/bin/activate
'''

## Install requirements
'''
pip install -r requirements.txt
'''

## Download the dataset and follow this directory structure
'''
NCTE_Transcripts/
├── classroom-transcript-analysis-main/
│   └── coding schemes/
├── LICENSE
├── README.md
├── requirements.txt
├── run_classifier.py
├── run_classifiers.sh
├── transcript_issues.txt
├── ncte_single_utterances
├── ncte_single_utterances.csv
├── paired_annotations.csv
├── student_reasoning.csv
└── transcript_metadata.csv
'''

## Do the initial clean and analysis of the dataset
'''
python src/initial_eval_and_clean.py
'''