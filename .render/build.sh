python -m pip install --upgrade pip
pip install -r requirements.txt
dvc remote modify remote-gdrive-github-actions --local \
      gdrive_user_credentials_file /etc/secrets/dvc-gdrive.json
dvc pull --remote remote-gdrive-github-actions