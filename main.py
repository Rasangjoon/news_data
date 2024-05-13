# 주요 함수 실행(데이터베이스 정보는 ~~ 으로 표시하였습니다., port 또한 1111로 임의의 숫자를 지정했습니다.)
import schedule
import time
from datetime import datetime, timedelta
from news_keywords_pipeline import NewsKeywordsPipeline

def job():
    # 현재 날짜
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # 이전 날짜(어제)
    yesterday = today - timedelta(days=1)
    # 매일 실행되는 파이프라인 작업
    news_keywords_pipeline = NewsKeywordsPipeline(configs={"host": "~~", "port": 0000, "user": "~~", "password": "~~", "database": "~~"})
    news_keywords_pipeline.run_pipeline(start_date=yesterday, end_date=today)

# 매일 정각에 job 함수 실행
schedule.every().day.at("00:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)