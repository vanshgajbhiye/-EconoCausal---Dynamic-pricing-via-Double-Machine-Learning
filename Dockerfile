# =============================================================================
# Stage 1: Build & Dependency Resolution Stage
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install minimal OS build tools required for compiling C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Pre-build binary wheels to eliminate build toolchain from final production image
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# =============================================================================
# Stage 2: Minimal Hardened Production Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runner

# Microservice runtime metadata
LABEL maintainer="Rohit Sahu <sahurohit16620@gmail.com>"
LABEL service="econocausal-pricing-api"
LABEL version="1.0.0"
LABEL description="Causal Dynamic Pricing API via Double Machine Learning"

# Set runtime optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ECONOCAUSAL_MODEL_PATH=/app/models/econocausal_pipeline.joblib \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Create non-privileged system group and user (security hardening)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Install curl for container healthchecking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder stage and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source code and serialized model artifacts
COPY src/ /app/src/
COPY models/ /app/models/
COPY data/raw/ /app/data/raw/
COPY README.md /app/README.md

# Ensure non-privileged ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser:appgroup

# Expose microservice REST port
EXPOSE 8000

# Docker native healthcheck hitting the FastAPI /health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server entrypoint
ENTRYPOINT ["uvicorn", "src.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log"]
