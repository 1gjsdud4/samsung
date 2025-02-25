import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import random
import time

# 삼성 브랜드 컬러 설정
PRIMARY_COLOR = "#1428A0"
BACKGROUND_COLOR = "#E6E6E6"

# 한글 폰트 설정
font_path = "font/NotoSansKR-Regular.ttf"  # 폰트 파일 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

def load_data():
    survey_path = "data/samsung_s25_sales_survey.xlsx"
    voc_path = "data/samsung_s25_education_voc_detailed.xlsx"
    
    survey_df = pd.read_excel(survey_path)
    voc_df = pd.read_excel(voc_path)
    
    return survey_df, voc_df

# 데이터 불러오기
survey_data, voc_data = load_data()

import streamlit as st
import time

@st.dialog("📚 필요한 교육 내용 분석", width="large")
def show_education_modal():
    st.markdown("### 📚 AI가 교육 내용을 분석 중...")

    # 로딩 애니메이션
    with st.status("🤖 AI가 교육 내용을 분석하는 중...", expanded=True) as status:
        time.sleep(2)  # 2초 동안 로딩 효과
        status.update(label="✅ 분석 완료!", state="complete")

    # 교육 내용 주제 리스트
    education_topics = [
        "실전 고객 응대 스크립트 및 FAQ 대응 방법",
        "AI 기능을 활용한 영업 전략",
        "갤럭시 S25 vs 경쟁 제품(아이폰 등) 비교 분석",
        "효율적인 온라인 세일즈 방법론",
        "영업 스킬 향상을 위한 시뮬레이션 교육",
        "매장 환경에서 활용할 수 있는 데모 및 실습 자료 제공",
        "제품의 차별점 강조 및 마케팅 전략"
    ]

    # 스트리밍 효과 적용
    st.markdown("### 📚 필요한 교육 내용 리스트")
    placeholder = st.empty()  # 빈 공간 생성
    full_text = ""  # 전체 텍스트를 저장할 변수

    for topic in education_topics:
        text = f"✅ {topic}"  # 체크 표시와 함께 한 줄 추가
        current_text = ""  # 현재 항목의 스트리밍용 변수

        for char in text:
            current_text += char  # 한 글자씩 추가
            placeholder.markdown(full_text + "\n\n" + current_text)  # 기존 항목 유지하며 업데이트
            time.sleep(0.05)  # 글자 하나당 0.05초 딜레이

        full_text += "\n\n" + current_text  # 스트리밍이 끝난 후 전체 텍스트에 추가
        time.sleep(0.3)  # 항목 간 약간의 딜레이 추가

    if st.button("교육 자료 만들기"):
        st.rerun()  # 모달을 닫기 위해 앱을 재실행




# Streamlit 페이지 설정
st.set_page_config(page_title="삼성 세일즈 교육 대시보드", layout="wide", page_icon="📊")

# 로고 추가
st.image("logo.png", width=100)
st.markdown(f"<h1 style='text-align: center; color: {PRIMARY_COLOR};'>삼성 세일즈 교육 대시보드</h1>", unsafe_allow_html=True)

if st.button("📚 필요 교육 내용 분석"):
    show_education_modal()



with st.container(): 


    # 화면 분할
    col0, col1, col2, col3 = st.columns([0.1, 1, 1, 0.1])

    with col1:
        st.header("📊 2월 4주차 교육 설문조사 결과")
        numeric_cols = survey_data.select_dtypes(include=['number']).columns
        survey_avg = survey_data[numeric_cols].mean()

        st.write("### ✅ 설문조사 평균 점수")
        
        # Streamlit의 dataframe 스타일 적용
        styled_df = survey_avg.to_frame(name="평균 점수").style.background_gradient(cmap="Blues").format("{:.2f}")

        # styled dataframe 표시
        st.dataframe(styled_df, use_container_width=True)
        
        # 📊 Streamlit 내장 바 차트 사용
        st.write("### 📊 설문조사 결과")
        st.bar_chart(survey_avg, use_container_width=True)
        

    with col2:
        st.header("📢 VOC(Voice of Customer) 분석")\

        # 🔍 특정 키워드 검색을 워드클라우드 위로 이동
        keyword = st.text_input("🔍 특정 키워드 검색")
        filtered_voc = voc_data[voc_data['VOC 내용'].str.contains(keyword, na=False)]

        st.markdown(f"### 🔎 '{keyword}' 관련 VOC")  
        st.dataframe(filtered_voc, height=400, use_container_width=True) 
        

        # 📌 더미 데이터 키워드 리스트
        dummy_keywords = [
            "S25", "S24", "차이점", "배터리", "수명", "카메라", "성능", "교육", "실습", "고객",
            "질문", "세일즈", "강사", "설명", "빠름", "이해", "어려움", "경쟁사", "아이폰", "샤오미",
            "비교", "분석", "가이드", "필요", "AI", "기능", "활용", "대응", "강조", "사례", "실전",
            "시간 배분", "개선", "신제품", "차별점", "세일즈 환경", "적용", "실무 중심", "교육 개선",
            "만족도", "향상", "고객 경험", "최적화", "마케팅 전략", "제품", "장점", "단점", "소비자",
            "응대", "효과", "초보자", "개념", "설명", "추가", "강조", "전략", "차별화"
        ]

        # 📌 VOC 데이터 확인 후 워드클라우드 생성
        text_data = " ".join(random.choices(dummy_keywords, k=200))  # 200개 단어 랜덤 생성

        # 📌 워드클라우드 생성 함수 (파란 계열 유지)
        def generate_wordcloud(text):
            wordcloud = WordCloud(
                font_path=font_path,
                width=500, height=400,
                background_color= '#FFFFFF' ,  # 사용자가 선택한 배경색 적용
                colormap="Blues"  # 파란 계열 고정
            ).generate(text)
            return wordcloud

        voc_wc = generate_wordcloud(text_data)

        # 🔥 워드클라우드 표시
        st.write("### 🎭 자주 언급된 키워드")
        st.image(voc_wc.to_array(), use_container_width=True)
        
