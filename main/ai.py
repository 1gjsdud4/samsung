import os
import time
import pandas as pd
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from langsmith import utils
from langchain.schema import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import streamlit as st
   
   
def run_major_recommendation():
    
    ######### 환경변수 설정 #########

 
    # os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    # os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    # os.environ["LANGCHAIN_TRACING"] = "true"
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    # os.environ["LANGCHAIN_PROJECT"] = "pr-only-making-30"

    load_dotenv()
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "pr-only-making-30"

    utils.tracing_is_enabled()


    ######### 그래프 정의 #########

    

    class State(TypedDict):
        survey_analysis: dict
        feedback_analysis: dict
        final_education_recommendation: dict
     

    graph_builder = StateGraph(State)



    ########## 노드 ###########

    class SurveyAnalysis(BaseModel):
        summary: str = Field(..., description="교육 프로그램 개선사항 요약")
        key_improvements: List[str] = Field(..., description="핵심 개선 제안 리스트")

    def survey(state: State):
        """
        설문조사 데이터를 분석하여 평균 점수 및 주요 개선 사항 도출
        """
        # CSV 파일 불러오기
        survey_file = "data/samsung_s25_sales_survey.csv"  # 설문조사 파일 경로
        df = pd.read_csv(survey_file, encoding="utf-8")
        
        df = df.head(15)

        # 점수형 문항(2~9번째 컬럼) 평균값 계산
        numeric_columns = df.columns[1:9]  # 두 번째 컬럼부터 점수형 데이터
        average_scores = df[numeric_columns].mean().to_dict()

        # 주관식 응답(가장 유용한 교육 내용 + 개선 제안) 합치기
        subjective_feedback = "\n".join(df.iloc[:, 9].dropna().tolist())  # 가장 유용했던 내용
        improvement_suggestions = "\n".join(df.iloc[:, 12].dropna().tolist())  # 개선 제안

        # Output Parser 설정 (JSON 형식)
        output_parser = JsonOutputParser(pydantic_object=SurveyAnalysis)
        format_instructions = output_parser.get_format_instructions()

        # 프롬프트 템플릿 정의
        prompt = PromptTemplate(
            template=f"""
            삼성 세일즈 교육 설문조사를 분석하고 주요 개선 방안을 제안해 주세요.

            ### 설문조사 평균 점수:
            {{average_scores}}

            ### 주관식 응답 (교육 내용 중 유용했던 부분):
            {{subjective_feedback}}

            ### 교육 개선을 위한 제안:
            {{improvement_suggestions}}

            위 데이터를 기반으로 *프로그램 개선 사항, 핵심 개선 제안 리스트**을 아래 JSON 형식으로 작성해 주세요.

            {{format_instructions}}
            """,
            input_variables=["average_scores","subjective_feedback", "improvement_suggestions","format_instructions"]
        )

        # LLM 모델 설정
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=1.0
        )

        # 프롬프트 체인 실행
        chain = prompt | llm | output_parser
        result = chain.invoke({
            "format_instructions": format_instructions,
            "average_scores": average_scores,
            "subjective_feedback": subjective_feedback,
            "improvement_suggestions": improvement_suggestions
        })

        return {"survey_analysis": result}

    class FeedbackAnalysis(BaseModel):
        summary: str = Field(..., description="설문조사 피드백의 요약")
        key_issues: List[str] = Field(..., description="주요 문제점 리스트")
        improvement_suggestions: List[str] = Field(..., description="개선 제안 리스트")


    def voc(state: State):

        # CSV 파일 불러오기
        feedback_file = "data/samsung_s25_education_voc_detailed.csv"  # 피드백 파일 경로
        df = pd.read_csv(feedback_file, encoding="utf-8", header=None)

        # 상위 15개 데이터만 사용
        df = df.head(15)

        # 첫 번째 열과 두 번째 열을 가져와 VOC 내용 구성
        feedback_text = "\n".join([
            f"{row.iloc[0]}: {row.iloc[1]}" for _, row in df.iterrows()
        ])
        print (feedback_text)
        # Output Parser 설정 (JSON 형식)
        output_parser = JsonOutputParser(pydantic_object=FeedbackAnalysis)
        format_instructions = output_parser.get_format_instructions()

        # 프롬프트 템플릿 정의
        prompt = PromptTemplate(
            template=f"""
            삼성 세일즈 교육 피드백을 분석하고 주요 문제점을 도출하여 개선 방안을 제안해 주세요.

            ### 교육 피드백:
            {{feedback_text}}

            위 데이터를 기반으로 **요약, 주요 문제점, 개선 제안**을 아래 JSON 형식으로 작성해 주세요.

            {{format_instructions}}
            """,
            input_variables=["feedback_text","format_instructions"]
        )

        # LLM 모델 설정
        llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=1.0
        )

        # 프롬프트 체인 실행
        chain = prompt | llm | output_parser
        result = chain.invoke({
            "format_instructions": format_instructions,
            "feedback_text": feedback_text
        })

        return {"feedback_analysis": result}


    class FinalEducationRecommendation(BaseModel):
        recommended_topics: List[str] = Field(..., description="필요한 교육 주제 리스트")
        justification: str = Field(..., description="이 교육 주제를 선정한 이유 요약")

    def final_education_recommendation_node(state: State):
        """
        설문조사 및 VOC 분석 결과를 바탕으로 필요한 교육 내용을 리스트업
        """
        # 설문조사 분석 결과 및 VOC 분석 결과 가져오기
        survey_analysis = state["survey_analysis"]
        feedback_analysis = state["feedback_analysis"]

        # Output Parser 설정 (JSON 형식)
        output_parser = JsonOutputParser(pydantic_object=FinalEducationRecommendation)
        format_instructions = output_parser.get_format_instructions()

        # 프롬프트 템플릿 정의
        prompt = PromptTemplate(
            template=f"""
            삼성 세일즈 교육을 위한 최적의 교육 내용을 도출해 주세요.

            ### 설문조사 분석 결과:
            {{survey_analysis}}

            ### VOC 피드백 분석 결과:
            {{feedback_analysis}}

            위 데이터를 종합하여, **필요한 교육 주제를 리스트업**하고, 
            **해당 주제를 선정한 이유**를 아래 JSON 형식으로 작성해 주세요.

            {{format_instructions}}
            """,
            input_variables=["survey_analysis","feedback_analysis","format_instructions"]
        )

        # LLM 모델 설정
        llm = ChatOpenAI(
            model='gpt-4o',
            temperature=1.0
        )

        # 프롬프트 체인 실행
        chain = prompt | llm | output_parser
        result = chain.invoke({
            "format_instructions": format_instructions, 
            "survey_analysis": survey_analysis,
            "feedback_analysis": feedback_analysis
        })

        return {"final_education_recommendation": result}


        
    ######### 노드 정의 ############

    graph_builder.add_node("설문 분석", survey)
    graph_builder.add_node("VOC 분석", voc)
    graph_builder.add_node("필요한 교육 내용 도출", final_education_recommendation_node)

    # ✅ 병렬 실행을 위해 `survey_analysis`와 `voc_analysis`를 동시에 실행
    graph_builder.add_edge(START, "설문 분석")
    graph_builder.add_edge(START, "VOC 분석")

    # ✅ 두 개의 분석 결과가 모두 완료되면 `final_education_recommendation` 실행
    graph_builder.add_edge("설문 분석", "필요한 교육 내용 도출")
    graph_builder.add_edge("VOC 분석", "필요한 교육 내용 도출")

    graph_builder.add_edge("필요한 교육 내용 도출", END)

    graph_training_recommendation = graph_builder.compile()

    # 그래프 시각화 저장
    graph_bytes = graph_training_recommendation.get_graph().draw_mermaid_png()
    with open('training_graph.png', 'wb') as f:
        f.write(graph_bytes)

######### 그래프 실행 ######### 

    state = {
        "survey_analysis": {}, 
        "feedback_analysis": {},
        "final_education_recommendation": {}
    }

    def graphstream(state):
        events = graph_training_recommendation.stream(
            state,
            stream_mode="messages"
        )
        print(f"[user]: {state["user_input"]}")
        print("[assistant]: ", end="")
        for event in events:
            message = event[0]
            if isinstance(message, AIMessage):
                print(message.content, end="", flush=True)
                time.sleep(0.05)
        print("\n")

    # graphstream(state)
    output = graph_training_recommendation.invoke(state)
    return output

# 예제 실행
if __name__ == "__main__":
    result = run_major_recommendation(
    )
    print(result)
