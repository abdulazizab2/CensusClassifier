# syntax = docker/dockerfile:1.2

FROM python:3.8-slim-buster

RUN apt-get update
RUN apt-get install git -y

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
#  render removes .git from docker
RUN git init
#  env var not activating unless using source
RUN chmod a+x .render/fetch_dvc.sh
RUN --mount=type=secret,id=dvc-gdrive.json,dst=/etc/secrets/dvc-gdrive.json
RUN /bin/bash -c "source ./.render/fetch_dvc.sh && dvc pull --remote remote-gdrive-github-actions"
EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0" ]