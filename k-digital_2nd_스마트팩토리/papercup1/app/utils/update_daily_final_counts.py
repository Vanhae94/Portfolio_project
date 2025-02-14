from datetime import datetime
from app import db
from app.models import standardLog, breakLog, NormalLog, DailyCount
import logging

def update_daily_final_counts():
    """AbnormalLog 및 NormalLog의 마지막 ID를 DailyCount 테이블에 저장"""

    today = datetime.today().strftime("%Y-%m-%d")

    try:
        last_standard = db.session.query(standardLog.id).order_by(standardLog.id.desc()).first()
        final_standard_count = last_standard[0] if last_standard else 0


        last_break = db.session.query(breakLog.id).order_by(breakLog.id.desc()).first()
        final_break_count = last_break[0] if last_break else 0

        last_normal = db.session.query(NormalLog.id).order_by(NormalLog.id.desc()).first()
        final_normal_count = last_normal[0] if last_normal else 0

        daily_record = DailyCount.query.filter_by(date=today).first()

        if daily_record:
            daily_record.final_standard_count = final_standard_count
            daily_record.final_break_count = final_break_count
            daily_record.final_normal_count = final_normal_count
        else:
            new_record = DailyCount(
                date=today,
                final_standard_count=final_standard_count,
                final_break_count=final_break_count,
                final_normal_count=final_normal_count
            )
            db.session.add(new_record)

        db.session.commit()
        logging.info(f"✅ DailyCount 업데이트 성공: {today}")
    
    except Exception as e:
        db.session.rollback()
        logging.error(f"🚨 DailyCount 업데이트 실패: {str(e)}")
