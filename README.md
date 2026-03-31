# ml_engine
# Forex ML GUI App

This project builds a machine learning pipeline for FX trading:

- Feature engineering
- Sliding-window walk-forward validation
- LightGBM ranking model
- PnL evaluation
- Future: Windows GUI app

## Structure

- notebooks/ → original notebooks
- src/ → Python modules (to be created)
- app/ → GUI app
- data/ → input/output data

## Next Steps

- Convert notebooks into Python modules
- Build GUI
- Package into Windows .exe



Used ChatGPT to create the notebooks that download data and run ML algorithms.  Used Colab to run them (I think I could do that from ChatGPT).
Created this GitHub repo and directories and populated thedata and notebooks folders.
Then used Codex to create Python code for the pieces of the notebooks.  Codex could import the notebooks from Github, but couldn't seem to commit the .py files back to GitHub
So I had to copy Codex's .py files manually and Upload them into Github into the /src folder, etc.
To run them, I had to use Colab again.

🔹 Step 1 — Open Colab
👉 https://colab.research.google.com
Click:
New Notebook
🔹 Step 2 — Clone your GitHub repo into Colab
In a cell:
!git clone https://github.com/tbillhar/ml_engine.git
%cd ml_engine
🔹 Step 3 — Install dependencies
Since Codex didn’t create requirements.txt, use this:
!pip install pandas numpy scikit-learn matplotlib lightgbm
🔹 Step 4 — Make sure your CSV is accessible
Upload your CSV manually:
Upload:
fx_features_wide.csv
🔹 Step 5 — Run your pipeline
!python main.py
