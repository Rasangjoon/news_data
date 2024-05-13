import pandas as pd
import numpy as np
import json
import re
import random
from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textrank import KeywordSummarizer
import pymysql
import plotly.express as px 
import matplotlib.pyplot as plt
# mysql을 사용하여 예시로 둔 autoDB 구축.
class Newskeywords:
    
    def __init__(self, configs) -> None:
        try:
            configs['port'] = int(configs.pop('port'))
            self.DB = pymysql.connect(**configs)
            print('데이터베이스 연결 성공')
        except pymysql.err.OperationalError as e:
            print("데이터베이스 연결 실패:", e)
        
        #형태소 분석기 초기화.
        self.okt = Okt()
        
        
    def __del__(self) -> None:
        # 데이터베이스 연결 해제
        self.DB.close()
        

    # 데이터 수집 함수(사전에 받은 json형식을 기반으로 수집하는 코드를 작성.)
    def collect_data(self, json_files):
        data_list = []

        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data_id = data["data_id"]
                for item in data["data_investing"]:
                    title = item["title"]
                    url = item["url"]
                    host = item["host"]
                    imgurl = item["imgurl"]
                    docsent = item["docsent"]
                    sentscore = item["sentscore"]
                    text = item["text"]
                    data_list.append({
                        "data_id": data_id,
                        "title": title,
                        "url": url,
                        "host": host,
                        "imgurl": imgurl,
                        "docsent": docsent,
                        "sentscore": sentscore,
                        "text": text
                    })
        df = pd.DataFrame(data_list)
        # text_str 열 추가
        df['text_str'] = df['text'].apply(lambda x: ' '.join(x))
        df.to_csv("combined_news.csv", index=False, encoding="utf-8")
        print("CSV 파일이 성공적으로 생성되었습니다.")

    def remove_special_characters(self, text):
        # 특수문자 제거를 위한 정규표현식
        text = re.sub(r"[^\w\s'|]", "", text)
        return text

    def rsc(self, text):
        # 특수문자 제거를 위한 정규표현식]
        text = re.sub(r"[^a-zA-Z0-9ㄱ-힣\s]", "", text)
        return text

    def remove_newline(self, text_list):
        # 리스트의 각 요소를 문자열로 결합
        text = ''.join(text_list)
        # 개행 문자 제거
        text = text.replace("\n", "")
        return text
    

    # 형태소 분석 및 토큰화 함수
    def tokenize(self, text):
        okt = Okt()
        # 형태소 분석 및 토큰화
        tokens = okt.morphs(text)
        return tokens


    def remove_stopwords(self, tokens):
        stopwords = ["을", "를", "이", "가", "은", "는", "이런", "저런",'가',',','이','다','는','에','기','지','을','가','로','0','고','의','한','하','2','대','1','은','인','자','사','시'
                            ,'를','해','서','원','수','도','상','정','업','전','으','장','보','제','3','스','했','부','리','"''',')','()','있','금','주','일','비','"','과','국','적',
                            '5','만','성','경','공','%','어','나','위','라','소','4','등','계','회','조','년','중','-','면','구','아','세','신','화','개','산','용','관','트','동','행',
                            '재','연','들','출','유','할','했다','에서','과','것','적','수','하는','하고','할','들','인','도','와','이다','해','있다','및','전','고','다','된','원','있는',
                            '말','까지','위','통해','기','3일','위해','대','중','1','제','부터','지','될','2','|','명','개','연','관련','이라고','더','된다','보다','주','이번','이상',
                            '에는','세','시','내','지난','으로','ㅣ','밝혔다','한다','됐다','에게','경우','따르면','이후','에도','액','대해','19','함께','예정','되는','그','점','하며',
                            '같은','최대','또한','따라','하면','대한','약','달','현재','주요','다양한','수준','했다고','때문','되고','에서는','지난해','대비','올해','간','총','날','전체',
                            '후','건','특히','가장','제공','기자','하기','지난달','하지','있도록','최근','분야','예상']

        tokens = [token for token in tokens if token not in stopwords]
        return tokens


    # 데이터 정제 함수
    def clean_data(self, df):
        df_cleaned = df.dropna(subset=['text'])
        df_cleaned['text'] = df_cleaned['text'].apply(self.remove_special_characters)
        df_cleaned['text'] = df_cleaned['text'].apply(self.rsc)
        df_cleaned['text'] = df_cleaned['text'].apply(self.remove_newline)
        df_cleaned['text'] = df_cleaned['text'].apply(self.tokenize)
        df_cleaned['text'] = df_cleaned['text'].apply(lambda x: self.remove_stopwords(x))
        return df_cleaned

    # TF-IDF 기반 핵심 키워드 추출 함수
    def extract_keywords_tfidf(self, df_cleaned):
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df_cleaned['text_str'])
        words = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.max(axis=0).toarray().flatten()
        word_tfidf_scores = list(zip(words, tfidf_scores))
        word_tfidf_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = 30
        top_keywords = [word for word, score in word_tfidf_scores[:top_n]]
        return top_keywords

    # TextRank 기반 핵심 키워드 추출 함수
    def extract_keywords_textrank(self, df_cleaned):
        text_data = df_cleaned['text'].apply(lambda x: ' '.join(x))
        summarizer = KeywordSummarizer(tokenize=self.tokenize, min_count=2, window=-1)
        keywords = summarizer.summarize(text_data.tolist(), topk=30)
        return keywords


    def random_color_func(self, word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
        return "rgb({}, {}, {})".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 워드 클라우드 생성 및 시각화 함수
    def generate_wordcloud(self, word_counts):
        wordcloud = WordCloud(
            font_path="NanumGothic.ttf",
            background_color="white",
            width=800,
            height=400,
            colormap="viridis",
            max_words=100,
            prefer_horizontal=0.7,
            min_font_size=10,
            max_font_size=200,
            random_state=42
        )
        wordcloud.generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud.recolor(color_func=self.random_color_func, random_state=3), interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
    def run_pipeline(self):
        #json 파일은 예시로 작성했습니다.
        json_files = ["20220204.json", "20220304.json", "20220404.json","20220504.json","20220604.json","20220704.json","20220804.json"]
        self.collect_data(json_files)
        df = pd.read_csv("combined_news.csv", encoding='utf-8')
        df_cleaned = self.clean_data(df)
        top_keywords_tfidf = self.extract_keywords_tfidf(df_cleaned)
        keywords_textrank = self.extract_keywords_textrank(df_cleaned)
        print("TF-IDF 기반 상위 키워드:")
        print(top_keywords_tfidf)
        print("TextRank 기반 상위 키워드:")
        print(keywords_textrank)


    
