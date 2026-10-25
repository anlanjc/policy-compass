FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY static ./static

EXPOSE 8000

# 首次启动自动增量摄取 data/corpus（含 18 篇演示政策）
CMD ["python", "scripts/serve.py"]
