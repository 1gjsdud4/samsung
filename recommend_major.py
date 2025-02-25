
import os
import time
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


######### 환경변수 설정 #########

load_dotenv()
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "pr-only-making-30"

utils.tracing_is_enabled()



######### 그래프 정의 #########

memory = MemorySaver()  

class Recommend_Major_State(TypedDict):
    user_input: str 
    index_name: str # pinecone index name
    vextorstore_name: str # pinecone vectorstore name
    retrieved_majors: list[dict[str, str]]
    retrived_count : int
    num_recommendations: int
    result : list[dict[str, str]]
    max_turn : int
    turn : int

graph_builder = StateGraph(Recommend_Major_State)



########## 노드 ###########

embedding_model_openai_3_small = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=os.getenv("OPENAI_API_KEY")
)



def retrive_major(state: Recommend_Major_State):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = state["index_name"]
    vectorstore_name = state["vextorstore_name"]
    
    host = pc.describe_index(index_name)["host"]
    index = pc.Index(host=host)

    
    vector = embedding_model_openai_3_small.embed_query(state["user_input"])
    
    rag_result = index.query(
        namespace=vectorstore_name,
        vector=vector,
        top_k=10,
        include_metadata=True
    )
    
    retrieved_majors = []
    for match in rag_result['matches']:
        major_name = match['metadata']['major_name']
        description = match['metadata']['description']
        retrieved_majors.append({"major": major_name, "description": description})
    
    return {"retrieved_majors": retrieved_majors}



class major_schma(BaseModel):
    recommended_major: str = Field(description="추천된 전공 이름")
    explanation: str = Field(description="추천 이유에 대한 간략한 설명")


class major_list(BaseModel):
    recommendations: List[major_schma] = Field(
        description="추천된 전공 리스트 (사용자가 원하는 개수만큼 조절 가능)"
    )

def chatbot_recommend(state: Recommend_Major_State):
    
    num_recommendations = state.get("num_recommendations", 3)
    
    output_parser = JsonOutputParser(pydantic_object=major_list)
    format_instructions = output_parser.get_format_instructions()
    
    #print(format_instructions)
    #print(state["retrieved_majors"])
    
    retrieved_info = "\n".join(
        [f"- 전공: {result['major']}, 설명: {result['description']}" for result in state["retrieved_majors"]]
    )
    
    prompt = PromptTemplate(
        
        template=f"""
        사용자 입력: {{query}}

        검색된 전공 정보:
        {retrieved_info}

        위 정보를 참고하여 **{num_recommendations}개의 추천 전공**을 선정하고, 
        각 전공의 추천 이유를 아래 JSON 형식으로 작성해 주세요.

        {{format_instructions}}
        """,
        input_variables=["query", "retrieved_info", "num_recommendations", "format_instructions"]
        
    )
    
    llm = ChatOpenAI(
        model= 'gpt-4o-mini',
        temperature= 1.0
    )
    
    chain = prompt | llm | output_parser
    result = chain.invoke({
        "query": state["user_input"],
        "retrieved_info": retrieved_info,
        "num_recommendations": num_recommendations,
        "format_instructions": format_instructions
    })
    
    print(result)
    return {"result": result}


def validate_recommendations(state: Recommend_Major_State):

    retrieved_majors = state.get("retrieved_majors", [])
    if state["turn"] < state["max_turn"]+1: 
        state["turn"] += 1
        
        if not retrieved_majors:
            print("⚠️ 검색된 전공 정보(retrieved_majors)가 없습니다.")
            return "retry_retrive_major"
        
        result = state.get("result", {}).get("recommendations", [])
        
        if not result:
            print("⚠️ 추천된 전공(result)이 없습니다.")
            return "retry_chatbot_recommend"


        
        school_majors = {major["major"] for major in retrieved_majors} 
        
        if len(school_majors) < state["num_recommendations"]:
            print("⚠️ 검색된 전공 수가 추천 전공 수보다 적음")
            state["num_recommendations"] += 5
            return "retry_retrive_major"
        
        recommended_majors = {rec["recommended_major"] for rec in result}

        missing_majors = recommended_majors - school_majors  # 차집합 연산으로 누락된 전공 찾기

        print("\n✅ 검색된 전공 리스트:", school_majors)
        print("✅ 추천된 전공 리스트:", recommended_majors)

        if missing_majors:
            print(f"⚠️ 추천된 전공 중 다음 전공이 학교 전공 리스트에 없음: {missing_majors}")
            return "retry_chatbot_recommend"
        
        print("✅ 모든 추천 전공이 학교 전공 리스트에 포함됨!")
        return "end"
    else:
        print("⚠️ 최대 턴 수 초과")
    
######### 노드 정의 ############

graph_builder.add_node("retrive_major", retrive_major)
graph_builder.add_node("chatbot_recommend", chatbot_recommend)


######### 엣지 정의 ###########3
graph_builder.add_edge(START, "retrive_major")
graph_builder.add_edge("retrive_major", "chatbot_recommend")
graph_builder.add_conditional_edges(
    "chatbot_recommend",
    validate_recommendations,
    {   
     "retry_retrive_major": "retrive_major",
     "retry_chatbot_recommend": "chatbot_recommend",
     "end": END
    }
)


graph_recommend_major = graph_builder.compile()

graph_bytes = graph_recommend_major.get_graph().draw_mermaid_png()

print("저장")
with open('graph.png', 'wb') as f:
    f.write(graph_bytes)
######### 그래프 실행 ######### 

state = {
    "user_input": "나는 창의적인 일과 사람을 돕는 것이 좋아. 어떤 전공이 나한테 맞을까?",
    "index_name": "chamajor",
    "vextorstore_name": "cha_major",
    "retrived_majors": [],
    "retrived_count": 10,
    "num_recommendations": 5,
    "result": [],
    "max_turn": 3,
    "turn": 0
}


def graphstream(state):
    #스트리밍 응답 방식
    events = graph_recommend_major.stream(
        state,
        stream_mode="messages"  
    )
    
    print(f"[user]: {state["user_input"]}")
    
    buffer = "" 

    print("[assistant]: ", end="")  

    for event in events:
        message = event[0] 
        
        if isinstance(message, AIMessage):
            buffer += message.content 
            print(message.content, end="", flush=True)  
            time.sleep(0.05)

    print("\n")

graphstream(state) 
output = graph_recommend_major.invoke(state)
print(output)