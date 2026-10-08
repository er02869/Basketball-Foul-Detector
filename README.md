# Capstone

My capstone and is a standalone browser application for uploading basketball video
and running it through the saved violation models. Predictions are experimental
until each model has been trained and validated.

## Start the standalone application

Install the dependencies, then run the browser application from this directory:

```bash
python -m pip install -r requirements.txt
python server.py
```

Open http://127.0.0.1:8502 in a browser. The application serves the JavaScript
frontend and Flask API together.
