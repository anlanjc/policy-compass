FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY static ./static

EXPOSE 8000

# 首次启动自动增量摄取 data/corpus（真实政策文件 + 演示文档，见 data/corpus 各文件头部来源声明）
CMD ["python", "scripts/serve.py"]
