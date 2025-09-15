FROM python:3.11-slim

WORKDIR /app

# instalar dependencias del sistema necesarias para psycopg2 y otros paquetes
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "FastApi:app", "--host", "0.0.0.0", "--port", "8000"]
