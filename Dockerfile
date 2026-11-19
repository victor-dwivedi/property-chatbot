FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN useradd -m -u 1000 app
WORKDIR /home/app/srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# copies app/ and, for local builds, data/ (on HF Spaces data is pulled from a private dataset at startup)
COPY --chown=app:app . .
RUN mkdir -p data && chown app:app data
USER app
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')" || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
