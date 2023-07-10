# syntax = docker/dockerfile:1.2

FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install git -y

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN --mount=type=secret,id=dvc-gdrive.json,dst=/etc/secrets/dvc-gdrive.json
# render removes .git from docker
RUN git init
RUN dvc remote modify remote-gdrive-github-actions --local gdrive_user_credentials_file dvc-gdrive.json
RUN dvc pull --remote remote-gdrive-github-action

EXPOSE 8000
CMD [ "uvicorn", "main:app" ]