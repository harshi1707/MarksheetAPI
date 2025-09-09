FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y poppler-utils build-essential \
&& pip install --no-cache-dir -r requirements.txt \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]