# Build from repository root: docker build -f apps/pfm/Dockerfile -t truffles-pfm .
FROM python:3.12-slim
WORKDIR /opt/pfm
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY apps/pfm/pyproject.toml apps/pfm/uv.lock ./apps/pfm/
COPY apps/pfm/src ./apps/pfm/src
RUN pip install --no-cache-dir uv==0.11.16 && \
    uv sync --project apps/pfm --frozen --no-dev --no-editable && \
    useradd --uid 10001 --create-home pfm && \
    mkdir -p /data && chown pfm:pfm /data
USER 10001:10001
ENV PFM_MODE=bank PFM_DATABASE=/data/pfm.sqlite3 PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["/opt/pfm/apps/pfm/.venv/bin/uvicorn", "truffles_pfm.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
