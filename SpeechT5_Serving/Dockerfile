FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchaudio

COPY . .

EXPOSE 5000

ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "app.py"]