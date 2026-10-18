# Run in Google Colab without uploading files

Colab starts with an empty computer every time. Instead of uploading the project, the notebooks **download it from GitHub** in their first cell
and download the data from Kaggle and NOAA. You only set this up once.

## One-time setup (about 10 minutes)

1. **Create a GitHub account** (github.com) and a **new repository** named `demand-forecasting-capstone` (empty, no README). Public is simplest.
2. **Put the project in it.** In a terminal inside the project folder:
   ```bash
   python scripts/set_repo_url.py https://github.com/<your-username>/demand-forecasting-capstone.git
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/demand-forecasting-capstone.git
   git push -u origin main
   ```
   (`set_repo_url.py` writes your repository address into every notebook.)
   No terminal? On github.com use *Add file -> Upload files* and drag in the project folder's contents, then run `set_repo_url.py` first.
3. **Open a notebook in Colab:** colab.research.google.com -> *File -> Open notebook -> GitHub* -> paste your repository address -> choose
   `notebooks/01_data_acquisition.ipynb`. Or click the "Open in Colab" badge at the top of any notebook on GitHub.
4. `Runtime -> Run all`. The first cell clones the repository and installs the package; nothing is uploaded.

## Private repository

Colab cannot clone a private repository without a key. Either make the repository public, or:
1. On GitHub: *Settings -> Developer settings -> Personal access tokens* -> create a token that can read the repository.
2. In Colab: the key icon (Secrets) -> add a secret `GITHUB_TOKEN` -> enable notebook access.
3. In the setup cell replace `REPO_URL` by
   ```python
   from google.colab import userdata
   REPO_URL = f"https://{userdata.get('GITHUB_TOKEN')}@github.com/<your-username>/demand-forecasting-capstone.git"
   ```
Never write the token into the notebook text.

## Alternative: Google Drive instead of GitHub

Copy the whole project folder to *My Drive/demand-forecasting-capstone*, then in the setup cell set `USE_DRIVE = True`. Colab asks once for permission to read your Drive.

## Real data inside Colab

Notebook 01 downloads from Kaggle. You need a Kaggle token (kaggle.com -> Settings -> API) and, for M5, to join the competition once.
The token is typed into a hidden box and is never saved in the notebook. Downloaded data lives only in that Colab session; run 01 and 02 again in a new session
(or save `data/processed/` to Drive to skip that).

## For the submission

Submit the GitHub repository link (the notebooks run from it) plus the final report. `reports/` holds the committed results.
