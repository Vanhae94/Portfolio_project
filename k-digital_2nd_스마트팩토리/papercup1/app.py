from flask import Flask
from app.routes.main import main_bp  # main.py에서 정의한 블루프린트
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
import os

# DB 객체 생성
db = SQLAlchemy()
migrate = Migrate()

def create_app():
    """Flask 앱을 생성하는 함수"""
    app = Flask(__name__)

    # 환경 변수에서 DB 설정 로드
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")

    # DB URI 설정
    app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # DB 초기화
    db.init_app(app)
    migrate.init_app(app, db)

    # 블루프린트 등록
    app.register_blueprint(main_bp)  # 'main' 블루프린트 등록
    
    return app
