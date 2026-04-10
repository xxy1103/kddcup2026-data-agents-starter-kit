FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DABENCH_INPUT_ROOT=/input \
    DABENCH_OUTPUT_ROOT=/output \
    DABENCH_LOG_ROOT=/logs \
    DABENCH_SUBMISSION_CONFIG=/app/configs/submission.yaml

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs/submission.yaml ./configs/submission.yaml

RUN mkdir -p /input /output /logs
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "dabench", "submit"]
