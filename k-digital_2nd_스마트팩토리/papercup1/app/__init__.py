import os
import logging
from logging.handlers import RotatingFileHandler  # 크기 제한을 위한 핸들러
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
import pymysql
import pandas as pd

# MySQL 드라이버 설정
pymysql.install_as_MySQLdb()

# 환경 변수 로드
load_dotenv()

# DB 객체 생성
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    """Flask 앱을 생성하는 함수"""
    app = Flask(__name__, template_folder='app/templates')  # 템플릿 폴더 명시적으로 설정

    # 환경 변수에서 DB 설정 로드
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
    DB_NAME = os.getenv("DB_NAME", "PaperCup")

    # SQLAlchemy 데이터베이스 URI 설정
    app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # DB 초기화
    db.init_app(app)
    migrate.init_app(app, db)

    # 로그 폴더 생성
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 크기 제한을 둔 로그 파일 핸들러 설정 (10MB 제한, 최대 3개의 백업 파일)
    log_file = os.path.join(log_dir, "app.log")
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.WARNING,  # 모든 로그 수준 기록
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[handler, logging.StreamHandler()]  # 파일 저장 및 콘솔 출력
    )

    # 블루프린트 등록
    from app.routes.main import main_bp  # 여기서 임포트
    app.register_blueprint(main_bp)

    return app
