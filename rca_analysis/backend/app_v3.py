from dataclasses import dataclass
from typing import Any, Tuple, Union, List, Dict
import streamlit as st
import httpx
import asyncio
import pandas as pd
from rca_prompts import *
from helper import *
import nest_asyncio
nest_asyncio.apply()
st.set_page_config(layout="wide")


st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: white;
        color: black;
        text-align: left;
    }
    button[kind="primary"]:hover {
        background: #ffc107;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
BTN_STATES_RCA = "btn_rca"
BTN_STATES_FAQ = "btn_faq"
RUNNING_JOBS = "running_jobs"
timeout = 3600
gen_url = "http://127.0.0.1:5000/genrca"
load_url = "http://127.0.0.1:5000/loadtask"
job_url = "http://127.0.0.1:5000/get_jobs"
available_cns_url = "http://127.0.0.1:5000/get_available_cases"


@dataclass
class Message:
    actor: str
    payload: str
    plot: bool
    sql: str



def initialize_session_state():
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I assist you?", plot=False, sql=None)]
        
    if BTN_STATES_RCA not in st.session_state:
        st.session_state[BTN_STATES_RCA] = {
            "btn8" : "Lack of understanding the customer's issue/case details",
            "btn9" : "Knowledge Gap / Poor Troubleshooting skills",
            "btn10" : "Inappropriate way of handle case",
            "btn11" : "Delayed response/follow-up",
            "btn12" : "Failure to send update within promised time",
            "btn13" : "Failure to set expectation",
            "btn14" : "Poor communication/conversational skills",
            "btn15" : "Difficult to contact Support",
            "btn16" : "Billing & Payments",
            "btn17" : "Privacy and TOS (Terms of service)",
            "btn18" : "Bug status",
            "btn19" : "Engineering/Specialist team intervention",
            "btn20" : "Complex / Lengthy troubleshooting process",
            "btn21" : "Customer Technical knowledge"
        }


    if BTN_STATES_FAQ not in st.session_state:
        st.session_state[BTN_STATES_FAQ] = {
            "btn1" : "Case Summary",
            "btn2" : "Case Sentiment",
            "btn3" : "Opportunity areas for the agent",
            "btn4" : "Case status change history",
            "btn5" : "Case priority change history",
            "btn6" : "Escalation reason",
            "btn7" : "Reason for DSAT",
            
        }

    if RUNNING_JOBS not in st.session_state:
        st.session_state[RUNNING_JOBS] = asyncio.run(fetch_running_jobs())


async def fetch_running_jobs() -> Union[List[Dict], str]:
    async with httpx.AsyncClient(timeout = timeout) as client:
        resp = await client.get(job_url)
        data = resp.json()["predictions"]
        return data
        

initialize_session_state()


def refresh():
    st.session_state[RUNNING_JOBS] = asyncio.run(fetch_running_jobs())


def extract_cn(query: str) -> Tuple[str, str]:
    idx = query.find("#")
    count = query.count("#")
    if(count > 1):
        return ("multi", "")
    else:
        cn = query[idx + 1: ].strip()[: 8]
        if cn.isnumeric():
            return ("ok", cn)
        else:
            return ("err", "")


async def generate_rca_from_rag(cn: str, prompt: str) -> str:
    query = {"prompt": prompt, "cn" : cn}
    async with httpx.AsyncClient(timeout = timeout) as client:
        resp = await client.post(gen_url, json = query)
        out = resp.json()["predictions"]
        return out



async def submit_job(cn):
    async with httpx.AsyncClient(timeout = timeout) as client:
        resp = await client.post(load_url, json = {"cn" : cn, "em" : "sagnikr"})
        if resp.status_code == 202:
            with cont:
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=f"Load job submitted. View log with JobID: '{resp.headers['Location']}'", plot=False, sql=None))
                with st.chat_message(ASSISTANT):
                    st.write(resp.headers["Location"])
            st.session_state.chat_inp = ""
            
        

@st.dialog("Please note")
def modal(err):
    st.markdown(f"### {err}")
    if st.button("Ok"):
        st.rerun()



@st.dialog("Please note")
def check_case(cn):
    st.markdown(f"### Cases {cn} not exist. Want to load ?")
    if st.button("Yes"):
        asyncio.run(submit_job(cn))
        st.rerun()


async def fetch_log(job_id):
    async with httpx.AsyncClient(timeout = timeout) as client:
        resp = await client.get(f"http://127.0.0.1:5000{job_id}")
        data = resp.json()
        return data
            

def view_log():
    with log_cont:
        try:
            out = asyncio.run(fetch_log(st.session_state.job_st))
        except Exception as e:
            out = "Please enter valid job id"
        st.write(out)


def clear_chat():
    st.session_state[MESSAGES] = st.session_state[MESSAGES][: 1]
    st.session_state.chat_inp = ""
    

_ = """def check_if_case_exists():
    cn_arr = list(map(str.strip, st.session_state.case_sch.split(',')))
    not_available = check_if_exists_case_arr(cn_arr)
    if not not_available:
        with view_cont:
            st.write(f"Case {st.session_state.case_sch} exists")
    else:
        available = list(set(cn_arr) - set(not_available))
        if available:
            with view_cont:
                st.write(f"Case {','.join(available)} exists")
        check_case(','.join(not_available))"""


async def handle_search_cases(cn_arr: List[str]) -> Any:
    async with httpx.AsyncClient(timeout = timeout) as client:
        resp = await client.post(available_cns_url, json = {"cn" : cn_arr})
        out = resp.json()["predictions"]
        return out



def check_if_case_exists():
    cn_arr = list(map(str.strip, st.session_state.case_sch.split(',')))
    out = asyncio.run(handle_search_cases(cn_arr))
    if isinstance(out, str):
        with view_cont:
            st.write(f"Something went wrong, please retry after sometime..")
    else:
        available, in_queue, in_bucket = out
        if not available and not in_queue:
            with view_cont:
                st.write(f"Cases **{st.session_state.case_sch}** exist.")
        elif not available and in_queue:
            with view_cont:
                if in_bucket:
                    st.write(f"Cases **{','.join(in_bucket)}** exist.")
                    st.write(f"Cases **{','.join(in_queue)}** already in queue.")
                else:
                    st.write(f"Cases **{','.join(in_queue)}** already in queue.")
        elif available and not in_queue:
            with view_cont:
                if in_bucket:
                    st.write(f"Cases **{','.join(in_bucket)}** exist.")
            check_case(','.join(available))
        else:
            with view_cont:
                if in_bucket:
                    st.write(f"Cases **{','.join(in_bucket)}** exist.")
                    st.write(f"Cases **{','.join(in_queue)}** already in queue.")
                else:
                    st.write(f"Cases **{','.join(in_queue)}** already in queue.")
            check_case(','.join(available))
            
                

def sample():
    cn = extract_cn(st.session_state.chat_inp)
    if (cn[0] == "ok"):
        if check_if_exists(cn[1]):
            with cont:
                st.session_state[MESSAGES].append(Message(actor=USER, payload=st.session_state.chat_inp, plot=False, sql=None))
                st.chat_message(USER).write(st.session_state.chat_inp)
                with st.spinner("Please wait.."):
                    out = asyncio.run(generate_rca_from_rag(cn[1], st.session_state.chat_inp))
                    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=out, plot=False, sql=None))
                    with st.chat_message(ASSISTANT):
                        st.write(out)
        else:
            check_case(cn[1])
        st.session_state.chat_inp = ""
    else:
        modal("Enter a valid case number in the prompt.")
    


column1, column2 = st.columns([0.65, 0.35])


with column1:
    msg: Message
    for msg in st.session_state[MESSAGES]:
        with st.chat_message(msg.actor):
            st.write(msg.payload)


    cont = st.container()
        
    st.write(" ")
    col_chat, col_btn2 = st.columns([0.9,0.1])
    with col_chat:
        with st.form("chat_form"):
            st.text_area(label = "Enter your query here", key = "chat_inp")
            st.form_submit_button('Ask', on_click = sample)
    with col_btn2:
        st.button("Clear", key="cl", on_click=clear_chat)


def case_faq_callback():
    q_init_rca = init_prompt('{CASE NUMBER}')
    if (st.session_state["btn1"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn1']}:\n  {q_summary} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn2"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn2']}:\n  {q_sentiment} {q_end}"
        st.session_state.chat_inp = prompt
    
    elif (st.session_state["btn3"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn3']}:\n  {q_opportunity} {q_end}"
        st.session_state.chat_inp = prompt
      
    elif (st.session_state["btn4"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn4']}:\n  {q_status_chng} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn5"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn5']}:\n  {q_priority_chng} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn6"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn6']}:\n  {q_esc_reason} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn7"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_FAQ]['btn7']}:\n  {q_dsat_reason} {q_end}"
        st.session_state.chat_inp = prompt

def rca_callback():
    q_init_rca = init_prompt('{CASE NUMBER}')
    if (st.session_state["btn8"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn8']}:\n  {q_issue_details} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn9"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn9']}:\n  {q_knowledge} {q_end}"
        st.session_state.chat_inp = prompt
    
    elif (st.session_state["btn10"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn10']}:\n  {q_transfer} {q_end}"
        st.session_state.chat_inp = prompt
      
    elif (st.session_state["btn11"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn11']}:\n  {q_delay_1} {q_delay_2} {q_delay_3} \n\n  {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn12"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn11']}:\n  {q_delay_4}  {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn13"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn13']}:\n  {q_expectation} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn14"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn14']}:\n  {q_comm} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn15"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn15']}:\n  {q_cont} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn16"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn16']}:\n  {q_billing} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn17"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn17']}:\n  {q_privacy} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn18"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn18']}:\n  {q_bug} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn19"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn19']}:\n  {q_internal} {q_end}"
        st.session_state.chat_inp = prompt
        
    elif (st.session_state["btn20"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn20']}:\n  {q_troubleshoot} {q_end}"
        st.session_state.chat_inp = prompt
            
    elif (st.session_state["btn21"]):
        prompt = f"{q_init_rca} 1. {st.session_state[BTN_STATES_RCA]['btn21']}:\n  {q_tech} {q_end}"
        st.session_state.chat_inp = prompt
        
    else:
        st.session_state.chat_inp = ""



with column2:
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>Check for available cases</h4>", unsafe_allow_html=True)
    with st.form("search_case"):
        st.text_input(label = "Search any case number", key = "case_sch")
        st.form_submit_button("View", on_click = check_if_case_exists)
        view_cont = st.container()

    with st.expander("View running jobs"):
        with st.form("log_form"):
            st.text_input(label = "Paste job id to view status", key = "job_st")
            st.form_submit_button("View", on_click = view_log)
            log_cont = st.container()
        st.write("")
        cp1, cp2 = st.columns([0.7, 0.3])
        with cp1:
            st.markdown("### Running Jobs")
        with cp2:
            st.button("Refresh", on_click = refresh)
        if isinstance(st.session_state[RUNNING_JOBS], str):
            st.error("Something wrong, please reload")
        else:
            if not st.session_state[RUNNING_JOBS]:
                st.info("No jobs running at this moment")
            else:
                df = pd.DataFrame(st.session_state[RUNNING_JOBS])
                st.dataframe(df, height = (len(df) + 1) * 35 + 3, hide_index = True, use_container_width=True)
        

    st.write("")
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>FAQ for case </h4>", unsafe_allow_html=True)
    col_faq1, col_faq2 = st.columns(2)
    for i in range(1, 8):
        if i % 2 != 0:
            with col_faq1:
                st.button(st.session_state[BTN_STATES_FAQ][f"btn{i}"], key = f"btn{i}", on_click = case_faq_callback, type="primary")
        else:
            with col_faq2:
                st.button(st.session_state[BTN_STATES_FAQ][f"btn{i}"], key = f"btn{i}", on_click = case_faq_callback, type="primary")
        
    st.markdown(f"<h4 style='text-align: center; font-weight: bold;'>RCA </h4>", unsafe_allow_html=True)
    col_rca1, col_rca2 = st.columns(2)
    for i in range(8, 22):
        if i % 2 != 0:
            with col_rca1:
                st.button(st.session_state[BTN_STATES_RCA][f"btn{i}"], key = f"btn{i}", on_click = rca_callback, type="primary")
        else:
            with col_rca2:
                st.button(st.session_state[BTN_STATES_RCA][f"btn{i}"], key = f"btn{i}", on_click = rca_callback, type="primary")

    st.write("")
    

        


