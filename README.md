## ⚙️ Installation

### Step 1: Install `uv`

First, install `uv` (if not already installed). `uv` is a faster and modern alternative to pip + virtualenv.

* Open the Command Prompt or Powershell
* Paste the given code into terminal

### Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

### Or use Homebrew (macOS):
brew install astral-sh/uv/uv

## 🔗 For more details: https://docs.astral.sh/uv/#installation

### Step 2: Create Virtual Environment & Install Dependencies

### a) Install the 'uv'
* pip install uv
  
### b) Create a virtual environment             
* uv venv

### c) Activate the virtual environment
* .venv/Scripts/activate         
* uv add -r requirements.txt

### Step 3: Run the Streamlit App

* streamlit run app.py

### Step 4: Create user_data.json in same directory along with app.py
* To store credentials in a local JSON file (not for production)


