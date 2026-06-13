cd "C:\Users\user\Downloads\NLP_P3"


# python -m venv env




Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


.\env\Scripts\Activate.ps1
# env\Scripts\activate.bat

python -m nltk.downloader stopwords

python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

$env:WANDB_API_KEY = "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"
# set WANDB_API_KEY="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"
python data/data_processing.py

$ErrorActionPreference = "Continue"

python main.py
