# Image multi-stage slim (Green IT, §5.6) — Debian slim permet l'install des wheels
# précompilés (scikit-learn/numpy/scipy n'ont pas de wheel musl/Alpine, ce qui
# forcerait une compilation source coûteuse). Résultat plus léger et build rapide.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app ./app
# Moteur interne uniquement : aucune exposition publique (routé par le MatchingService).
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
