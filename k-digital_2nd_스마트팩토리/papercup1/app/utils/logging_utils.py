import os
import logging
import pandas as pd

def log_message(message):
    """메시지가 비어있지 않으면 로그를 기록하는 함수"""
    if message.strip():  # 메시지가 비어있지 않으면
        logging.info(message)
    else:
        logging.debug("Empty message, not logging.")  # 비어있는 메시지에 대한 로깅 (필요시)

def save_logs_to_excel():
    """로그 파일을 읽어 엑셀로 저장하는 함수"""
    log_file = os.path.join("logs", "app.log")
    log_entries = []
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            for line in file:
                # 공백 줄 또는 형식이 잘못된 줄은 무시
                if not line.strip():
                    continue
                # 로그 라인이 올바른 형식인지 확인하고 파싱
                parts = line.split(" - ", 2)
                if len(parts) == 3:
                    timestamp, level, message = parts
                    log_entries.append([timestamp, level, message.strip()])
                else:
                    logging.warning(f"형식에 맞지 않는 로그 라인: {line.strip()}")

        # pandas DataFrame으로 변환
        df_logs = pd.DataFrame(log_entries, columns=["Timestamp", "Level", "Message"])

        # 엑셀 파일로 저장
        excel_file = os.path.join("logs", "log_data.xlsx")
        df_logs.to_excel(excel_file, index=False)

        logging.info(f"로그 파일이 엑셀로 저장되었습니다: {excel_file}")

    except Exception as e:
        logging.error(f"로그 파일을 엑셀로 저장하는 중 오류 발생: {e}")
