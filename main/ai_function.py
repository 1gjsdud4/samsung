import os
import time
from dotenv import load_dotenv
import pandas as pd


from langsmith import utils

from langchain.schema import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

######### 환경변수 설정 #########
load_dotenv()
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "pr-only-making-30"

utils.tracing_is_enabled()

######### 그래프 정의 #########

class TrainingRecommendationState(TypedDict):

    survey_results: str
    voc_results: str
    survey_analysis: str
    voc_analysis: str
    final_analysis: str


graph_builder = StateGraph(TrainingRecommendationState)

########## 노드 정의 ###########


import pandas as pd

def load_excel_data_as_text():
    """
    CSV 파일을 불러와서 텍스트(str) 형식으로 변환
    """
 
    voc_file = "data/samsung_s25_education_voc_detailed.csv"
    
    def format_survey_data():
        """
        CSV 파일을 읽어 텍스트 형식의 프롬프트로 변환
        """
        survey_file = "data/samsung_s25_sales_survey.csv"  # 파일 경로

        # CSV 파일 읽기
        df = pd.read_csv(survey_file, encoding="utf-8")
        
        df = df.head(10)

        # 문장으로 변환
        prompt_text = "다음은 삼성 세일즈 교육에 대한 설문조사 결과입니다:\n\n"

        for _, row in df.iterrows():
            prompt_text += f"사용자 {row['사용자 아이디']}:\n"
            prompt_text += f"- 교육 주제가 현재 업무와 관련이 있었는가? {row.iloc[1]}/5\n"
            prompt_text += f"- 교육 내용이 실제 영업에 적용하기 쉬웠는가? {row.iloc[2]}/5\n"
            prompt_text += f"- 교육을 통해 영업 스킬이 향상되었는가? {row.iloc[3]}/5\n"
            prompt_text += f"- 교육 후 영업 업무에 대한 자신감이 증가했는가? {row.iloc[4]}/5\n"
            prompt_text += f"- 온라인 학습 플랫폼 사용이 편리했는가? {row.iloc[5]}/5\n"
            prompt_text += f"- 교육 자료(동영상, PDF 등)의 품질은 어떠했는가? {row.iloc[6]}/5\n"
            prompt_text += f"- 교육 중 다른 참가자들과 상호작용 기회가 충분했는가? {row.iloc[7]}/5\n"
            prompt_text += f"- 교육에서 배운 내용을 실제 업무에 적용할 수 있는가? {row.iloc[8]}/5\n"
            prompt_text += f"- 가장 유용했던 교육 내용: {row.iloc[9]}\n"
            prompt_text += f"- 전반적인 교육 만족도: {row.iloc[10]}/5\n"
            prompt_text += f"- 동료에게 추천할 의향이 있는가? {row.iloc[11]}/5\n"
            prompt_text += f"- 교육 개선을 위한 제안: {row.iloc[12]}\n\n"

        return prompt_text

    # CSV 파일 읽기
    voc_df = pd.read_csv(voc_file, encoding="utf-8")

    # 데이터프레임을 문자열로 변환 (탭 구분)
    survey_text = format_survey_data()
    voc_text = voc_df.to_csv(index=False, sep="\t", encoding="utf-8")

    # str 타입으로 반환
    return survey_text, str(voc_text)


# 1️⃣ 설문조사 분석 노드
def survey_analysis_node(state: TrainingRecommendationState):
    """
    사용자 설문 데이터를 바탕으로 교육 필요 요소 분석
    """
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=1.0)
    print(state["survey_results"])
    
    prompt = PromptTemplate(
        
        template=f"""
        사용자 설문조사 결과:
        {{results}}

        설문내용을 분석해서 정리해주세요.
        """,
        input_variables=["results"]
    )

    
    chain = prompt | llm 
    result = chain.invoke({
        "results": state["survey_results"]
    })
    
    return {"survey_analysis": result}

# 2️⃣ VOC 분석 노드
def voc_analysis_node(state: TrainingRecommendationState):

    print(state["voc_results"])
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=1.0)
    
    prompt = PromptTemplate(
        template=f"""
        VOC 분석 데이터:
        {{voc_results}}

        설문내용을 분석해서 정리해주세요.
        """,
        input_variables=["voc_results"]
    )
    
    chain = prompt | llm 
    result = chain.invoke({
        "voc_results": state["voc_results"]
    })
    
    return {"voc_analysis": result}

# 3️⃣ 최종 교육 내용 추천 노드
def final_education_recommendation_node(state: TrainingRecommendationState):
    """
    설문조사와 VOC 데이터를 결합하여 최종 교육 내용 추천
    """
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=1.0)
    
    prompt = PromptTemplate(
        template="""
        설문조사 분석 결과:
        {{survey_analysis}}

        VOC 분석 결과:
        {{voc_analysis}}

        위 데이터를 종합하여 삼성 세일즈 교육에서 제공해야 할 필수 교육 내용을 정리해 주세요.
        """,
        input_variables=["survey_analysis", "voc_analysis"]
    )

    # output_parser = JsonOutputParser()
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "survey_analysis": state["survey_analysis"],
        "voc_analysis": state["voc_analysis"]
    })
    
    return {"final_analysis": result}

######### 그래프 정의 (병렬 실행) #########
graph_builder.add_node("설문 분석", survey_analysis_node)
graph_builder.add_node("VOC 분석", voc_analysis_node)
graph_builder.add_node("final_education_recommendation", final_education_recommendation_node)

# ✅ 병렬 실행을 위해 `survey_analysis`와 `voc_analysis`를 동시에 실행
graph_builder.add_edge(START, "설문 분석")
graph_builder.add_edge(START, "VOC 분석")

# ✅ 두 개의 분석 결과가 모두 완료되면 `final_education_recommendation` 실행
graph_builder.add_edge("설문 분석", "final_education_recommendation")
graph_builder.add_edge("VOC 분석", "final_education_recommendation")

graph_builder.add_edge("final_education_recommendation", END)

# 그래프 컴파일
graph_training_recommendation = graph_builder.compile()

# 그래프 시각화 저장
graph_bytes = graph_training_recommendation.get_graph().draw_mermaid_png()
with open('training_graph.png', 'wb') as f:
    f.write(graph_bytes)

######### 그래프 실행 #########

survey_text, voc_text = load_excel_data_as_text()


state = {
    "user_input": "",
    "survey_results": "안녕",
    "voc_results": voc_text,
    "survey_analysis": "",
    "voc_analysis": "",
    "final_analysis": "",

}

def graphstream(state):
    """
    스트리밍 방식으로 AI 응답을 출력
    """
    events = graph_training_recommendation.stream(
        state,
        stream_mode="messages"
    )
    print("[assistant]: ", end="")

    for event in events:
        message = event[0] 
        
        if isinstance(message, AIMessage):
            buffer += message.content 
            print(message.content, end="", flush=True)  
            time.sleep(0.05)

    print("\n")

print()
graphstream(state)