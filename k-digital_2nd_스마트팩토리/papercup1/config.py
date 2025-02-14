import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# MySQL 데이터베이스 설정
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_NAME = os.getenv("DB_NAME", "PaperCup")

# Flask 앱 설정
FLASK_SECRET_KEY = os.urandom(24)
FLASK_DEBUG = os.getenv("DEBUG", "True") == "True"

# 데이터베이스 URI (환경 변수 DATABASE_URL을 사용)
SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
SQLALCHEMY_TRACK_MODIFICATIONS = False
