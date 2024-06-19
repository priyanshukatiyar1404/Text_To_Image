pip install --upgrade pip
pip install virtualenv
python -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run main.py
