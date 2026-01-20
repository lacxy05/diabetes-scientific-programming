# Diabetes Prediction – Scientific Programming Final Project

## Project Overview
This project develops a machine learning pipeline to predict diabetes using clinical data.
The workflow includes data preprocessing, model training, evaluation, API deployment, and a simple user interface.

The project is developed collaboratively as part of the *Scientific Programming* course, with each team member responsible for a specific component of the pipeline and working on an individual Git branch.

---

## Project Structure

```
diabetes-scientific-programming/
│
├── data/          # Dataset (diabetes.csv)
├── models/        # Trained model and scaler
├── plots/         # EDA and evaluation figures
├── src/           # Preprocessing, training, evaluation, UI
├── api/           # FastAPI application
│
├── config.py      # Shared configuration variables
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/lacxy05/diabetes-scientific-programming.git
cd diabetes-scientific-programming
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
The full list of dependencies will be provided in `requirements.txt`.
```bash
pip install -r requirements.txt
```

---

## Project Workflow

The project is designed to be run in the following order:

1. **Data preprocessing and exploratory data analysis**
```bash
python src/preprocess.py
```

2. **Model training**
```bash
python src/train.py
```

3. **Model evaluation**
```bash
python src/evaluate.py
```

4. **Run the API**
```bash
uvicorn api.app:app --reload
```

5. **Run the user interface**
```bash
streamlit run src/dashboard.py
```

---

## Configuration
Shared paths and configuration variables are defined in `config.py`.

All scripts should import paths from this file rather than hard-coding them.

Example:
```python
from config import DATA_PATH, MODELS_DIR
```

---

## Collaboration and Git Workflow

- Each collaborator works on their assigned feature branch.
- Python files are created and developed by the person responsible for that task.
- Pull Requests are reviewed and merged into `main`.
- The `main` branch should always remain stable and runnable.

---

## Notes
- Generated files such as trained models (`.pkl`) are not tracked by Git.
- Figures required for the report are saved in the `plots/` directory.
- Docker and deployment instructions will be added in a later stage of the project.

---

## Authors
Scientific Programming – Master’s in Health Data Science
