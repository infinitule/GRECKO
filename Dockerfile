# GRECKO C2 bridge — deployable decision-engine service.
# Simulation & C2 software only; no hardware control, no weapon integration.
FROM python:3.11-slim AS base

# Non-root runtime user
RUN useradd --create-home --uid 10001 grecko

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
COPY grecko ./grecko
COPY sim ./sim
COPY league ./league
COPY s2r ./s2r
COPY eval ./eval
COPY learn ./learn
COPY tools ./tools
COPY proto ./proto

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

USER grecko

ENV GRECKO_HOST=0.0.0.0 \
    GRECKO_PORT=8765 \
    GRECKO_SEED=42 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

# Liveness: the bridge port is accepting TCP connections.
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import socket,os; s=socket.create_connection(('127.0.0.1', int(os.environ['GRECKO_PORT'])), 3); s.close()" || exit 1

# sh -c so the env vars expand at runtime
CMD ["sh", "-c", "grecko serve --host \"$GRECKO_HOST\" --port \"$GRECKO_PORT\" --seed \"$GRECKO_SEED\""]
