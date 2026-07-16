FROM python:3.12-slim

LABEL org.opencontainers.image.title="PerfoSim Pro v3"
LABEL org.opencontainers.image.description="Niger Delta Perforation & Hydraulic Fracturing Optimization Framework"

WORKDIR /app

# System deps: matplotlib/reportlab need a few font/render libs; curl is
# needed by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6 libpng16-16 fonts-liberation curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
