# ----------------------------------------------
# Dockerfile for RAG-CFT
# ----------------------------------------------
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --group raguser

# Copy and install depedendencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Change ownership of the app directory
RUN chown -R raguser:appgroup $APP_HOME

# Switch to the non-root user
USER raguser

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]