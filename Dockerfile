# syntax=docker/dockerfile:1
#
# Stages:
#   dev         uv and every dependency group. docker-compose.yaml and
#               docker-compose.test.yaml mount the working tree over /app.
#   production  the app and its runtime dependencies only, as a non-root
#               user. Released to ghcr.io by .github/workflows/release.yaml.
#
# The python tag matches .python-version and requires-python in
# pyproject.toml; change all three together.

FROM python:3.15-slim-trixie AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/venv \
    PATH=/venv/bin:$PATH
WORKDIR /app

FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
# Dependencies first, so a source change doesn't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM build AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
CMD ["python", "-m", "slack2rag"]

FROM base AS production
# Numeric, so orchestrators can verify it isn't root. /data holds the sync
# state (STATE_FILE) and /results the EVAL_TEST output; both are volumes
# the app user must be able to write.
RUN groupadd --system --gid 10001 app \
 && useradd --system --gid 10001 --uid 10001 --no-log-init --no-create-home app \
 && mkdir /data /results \
 && chown 10001:10001 /data /results
COPY --from=build /venv /venv
USER 10001:10001
CMD ["slack2rag"]
