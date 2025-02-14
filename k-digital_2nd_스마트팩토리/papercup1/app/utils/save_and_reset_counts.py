from flask import current_app
from app.models import standardLog, breakLog, NormalLog, DailyCount
from datetime import date, datetime
from app import db
import logging

# ✅ 전역 변수 선언
LAST_DATE = None
Standard_detected = 0
break_detected = 0
normal_detected = 0

# ✅ 로깅 설정
logger = logging.getLogger(__name__)

def save_and_reset_counts():
    """하루 동안의 로그 개수를 DailyCount에 저장하고 로그 초기화"""
    global Standard_detected, break_detected, normal_detected, LAST_DATE
    
    with current_app.app_context():
        today = date.today()

        # ✅ 날짜 변경 감지 (하루가 지났을 경우)
        if LAST_DATE is None or today > LAST_DATE:
            LAST_DATE = today  # 날짜 업데이트

            try:
                # ✅ 하루 동안의 최종 개수 계산
                Standard_count = standardLog.query.count()
                break_count = breakLog.query.count()
                normal_count = NormalLog.query.count()

                # ✅ DailyCount에 저장 (날짜가 이미 있으면 업데이트)
                daily_record = DailyCount.query.filter_by(date=LAST_DATE).first()
                if daily_record:
                    daily_record.final_standard_count = Standard_count
                    daily_record.final_break_count = break_count
                    daily_record.final_normal_count = normal_count
                    logging.info(f"📌 [업데이트] {LAST_DATE} - 규격 불량: {Standard_count}, 파손 불량: {break_count}, 정상: {normal_count}")
                else:
                    new_daily_count = DailyCount(
                        date=LAST_DATE,
                        final_standard_count=Standard_count,
                        final_break_count=break_count,
                        final_normal_count=normal_count
                    )
                    db.session.add(new_daily_count)
                    logging.info(f"✅ [신규 저장] {LAST_DATE} - 규격 불량: {Standard_count}, 파손 불량: {break_count}, 정상: {normal_count}")

                db.session.commit()

                # ✅ 로그 테이블 초기화 (모든 데이터 삭제)
                db.session.query(standardLog).delete()
                db.session.query(breakLog).delete()
                db.session.query(NormalLog).delete()
                db.session.commit()
                logging.info(f"🗑️ {LAST_DATE} 로그 테이블 초기화 완료")

                # ✅ 카운트 초기화
                Standard_detected = 0
                break_detected = 0
                normal_detected = 0

                logging.info(f"✅ [{datetime.now()}] DailyCount 저장 완료 및 로그 초기화 완료.")

            except Exception as e:
                db.session.rollback()
                logging.error(f"🚨 [{datetime.now()}] DailyCount 저장 실패: {str(e)}")
