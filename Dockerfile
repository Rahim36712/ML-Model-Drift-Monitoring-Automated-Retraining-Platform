FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLOPS_HOME=/workspace/data/demo API_HOST=0.0.0.0 DASHBOARD_HOST=0.0.0.0
WORKDIR /workspace
COPY requirements.lock ./
RUN pip install --no-cache-dir --timeout 120 -r requirements.lock
COPY . .
RUN useradd --create-home app && mkdir -p /workspace/data/demo && chown -R app:app /workspace
USER app
EXPOSE 8000 8050 5000
HEALTHCHECK --interval=15s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=5)"
CMD ["python", "-m", "scripts.run_demo", "--no-browser"]
