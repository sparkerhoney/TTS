FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 라이브러리 설치를 위한 파일 복사
COPY requirements.txt .

# 라이브러리 설치
RUN pip install -U pip && \
    pip install -r requirements.txt

# Flask 애플리케이션 파일 복사
COPY . .

# Flask 애플리케이션 실행
CMD ["flask", "run", "--host=0.0.0.0"]
