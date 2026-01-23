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
├── data/                                    # Dataset and data splits
│   ├── diabetes.csv                         # Original dataset
│   ├── diabetes_cleaned.csv                 # Cleaned dataset
│   ├── dataset_description.txt              # Dataset documentation
│   ├── train_split.csv                      # Training data
│   └── test_split.csv                       # Testing data
│
├── models/                                  # Trained models and scalers
│
├── plots/                                   # EDA and evaluation figures
│
├── src/                                     # Source code and notebooks
│   ├── __init__.py
│   ├── 00_SP_project_EDA.ipynb              # Exploratory Data Analysis
│   ├── 01_preprocess.py                     # Data preprocessing
│   ├── 02_traintest_split.py                # Train-test splitting
│   ├── 03_train.py                          # Model training
│   ├── 04_SP_project_Evaluation_with_balanced.ipynb  # Model evaluation
│   ├── 05_Quality_Checks.ipynb              # Code quality checks
│   └── 06_dashboard.py                      # Streamlit dashboard
│
├── api/                                     # FastAPI application
│   └── app.py                               # API endpoints
│
├── requirements.txt                         # Python dependencies
├── Dockerfile                               # Docker configuration
├── entrypoint.sh                            # Docker entrypoint script
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore rules
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
```bash
pip install -r requirements.txt
```

---

## Docker Deployment

### Build and run with Docker
```bash
# Build the Docker image
docker build -t diabetes-prediction .

# Run the container
docker run -p 8000:8000 -p 8501:8501 diabetes-prediction
```

The application will be accessible at:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## Quick Start

**Note:** This repository contains the final version of the project with all generated files already included (trained models, plots, cleaned datasets, and data splits). If you just want to try the application, you can skip the entire pipeline and jump directly to step 7 or 8 below.

---

## Project Workflow

The project is designed to be run in the following order:

### 1. Exploratory Data Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook src/00_SP_project_EDA.ipynb
```

### 2. Data preprocessing
```bash
python src/01_preprocess.py
```

### 3. Train-test split
```bash
python src/02_traintest_split.py
```

### 4. Model training
```bash
python src/03_train.py
```

### 5. Model evaluation
Open and run the evaluation notebook:
```bash
jupyter notebook src/04_SP_project_Evaluation_with_balanced.ipynb
```

### 6. Quality checks
Open and run the quality checks notebook:
```bash
jupyter notebook src/05_Quality_Checks.ipynb
```

### 7. Run the API
```bash
uvicorn api.app:app --reload
```
The API will be available at http://localhost:8000

### 8. Run the dashboard
```bash
streamlit run src/06_dashboard.py
```
The dashboard will be available at http://localhost:8501

---

## Configuration
Shared paths and configuration variables are defined in the project scripts.

---

## Collaboration and Git Workflow

- Each collaborator of the group worked on their assigned feature branch
- Additional branches were created for fixes and improvements as needed
- Python files have been created and developed by the person responsible for that task
- Pull Requests have been reviewed and merged into `main`
---

## Technologies Used

- **Python 3.x**: Core programming language
- **scikit-learn**: Machine learning library
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Data visualization
- **FastAPI**: REST API framework
- **Streamlit**: Dashboard framework
- **Docker**: Containerization
- **Jupyter**: Interactive notebooks

---

## Notes
- This repository includes all generated files (`.pkl` models, `.csv` data splits, and plots)
- Users can run the application directly without executing the entire pipeline
- Figures required for the report are saved in the `plots/` directory
- The project is fully containerized and can be deployed using Docker

---

## Authors
Scientific Programming – Master’s in Health Data Science
Laia Colomé Xicoy, Joana Ros Alonso, Rajae el Gaouzi, João Constantino Muianga, Carla Bellido García, Michael George Nabih Lotfy Eskander, Yuxi Qiao