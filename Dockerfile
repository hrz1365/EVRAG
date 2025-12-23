# ============================================================
# 1. Base image
# ============================================================
FROM python:3.11-slim

# ============================================================
# 2. Python runtime optimizations
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1       
ENV PYTHONUNBUFFERED=1              
ENV TRANSFORMERS_CACHE=/app/models  

# ============================================================
# 3. Configurable non-root user setup (best practice)
# ============================================================
ARG USERNAME=raguser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# ============================================================
# 4. System dependencies (only what's needed)
# ============================================================
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 5. Create app directories with correct permissions
# ============================================================
WORKDIR /app
RUN mkdir -p /app/models /app/data /app/reports && \
    chown -R ${USERNAME}:${USERNAME} /app

# ============================================================
# 6. Install Python dependencies first (for caching)
# ============================================================
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ============================================================
# 7. Copy your application code
# ============================================================
COPY . /app
RUN chown -R ${USERNAME}:${USERNAME} /app

# ============================================================
# 8. Switch to non-root user (security best practice)
# ============================================================
USER ${USERNAME}

# ============================================================
# 9. Expose port (for API or Streamlit later)
# ============================================================
EXPOSE 8501

# ============================================================
# 10. Default CMD (overridden by docker-compose if needed)
# ============================================================
CMD ["python", "main.py"]