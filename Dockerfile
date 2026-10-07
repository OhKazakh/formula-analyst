FROM python:3.13-slim

# Offline mode serves only the bundled data, the same way the deployed app runs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    FORMULA_ANALYST_OFFLINE=1

RUN useradd --create-home app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# The app only writes FastF1's cache; everything else stays read-only for it.
RUN install -d -o app -g app cache
USER app

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py"]
