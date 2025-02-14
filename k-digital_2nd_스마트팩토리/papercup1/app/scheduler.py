from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.save_and_reset_counts import save_and_reset_counts

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(save_and_reset_counts, "cron", hour=0, minute=0)  # 매일 자정 실행
    scheduler.start()
