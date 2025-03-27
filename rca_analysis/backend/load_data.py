from dataclasses import dataclass, field, asdict
from typing import TypedDict, Union, Literal, Generic, TypeVar,Any, Union, cast, Type, List, Tuple, Any
import pandas as pd
#from count_tokens.count import count_tokens_in_string
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime
from bs4 import BeautifulSoup
import re

creds_bq = service_account.Credentials.from_service_account_file(
                "topaz-poc-2024-new1.json",scopes=['https://www.googleapis.com/auth/cloud-platform',
                              "https://www.googleapis.com/auth/drive",
                              "https://www.googleapis.com/auth/bigquery",])

client_bq = bigquery.Client(credentials = creds_bq, project = "topaz-poc-2024")


def extract_data_from_cn(cn: int) -> pd.DataFrame:
    query = f"""
        select * from `topaz-poc-2024.agents_casedata.gcp_case_feeds_closed` where case_number = @cn order by timestamp asc; 
    """
    query_params=[
        bigquery.ScalarQueryParameter("cn", "INTEGER", cn),
    ]

    job_config=bigquery.QueryJobConfig(
        query_parameters=query_params
    )

    return client_bq.query(query,job_config=job_config).result().to_dataframe()


def break_chunks(data_arr: List[str], token_size: int = 150) -> str:
    s = ""
    inner = ""
    for i, j in enumerate(data_arr):
        inner += j + "\n"
        count_tok = count_tokens_in_string(inner)
        if(count_tok > token_size):
            s += f"{inner}</S_T>"
            inner = ""

        if(inner != "" and i == len(data_arr) - 1):
            s += f"{inner}</S_T>"
            
    return s


def break_chunks_v2(data_arr: List[str]) -> str:
    out = ""
    for i,j in enumerate(data_arr):
        out += f"A.{i+1}: {j}\n"
    return out


def preProcess(msg):
    sent = msg.split("\n")

    out_sent = ""

    for i in range(len(sent)):
        if sent[i].strip().startswith('From:'):
            break
            
        if re.search("Google Cloud Support <EMAIL_ADDRESS>".lower(), sent[i]):
            break

        if re.search("CONFIDENTIALITY NOTICE:".lower(), sent[i]):
            break

        if re.search("Google Cloud Support, <EMAIL_ADDRESS>".lower(), sent[i]):
            break

        if re.search("The information contained in this transmission may contain privileged".lower(), sent[i]):
            break

        if re.search("This message contains information that is confidential".lower(), sent[i]):
            break

        out_sent += sent[i] + "\n"

    return out_sent

def preprocess_block_chat_transcript(text: str) -> str:
    if bool(BeautifulSoup(text, "html.parser").find()):
        tag = '<p align="center">'
        iterable = list(re.finditer(tag, text))
        if not iterable:
            return ""
        else:
            end_idx = iterable[-1].span()[1]
            text = text[end_idx : ]

    c_str = ":"
    c_idx = text.find(c_str)
    text = text[c_idx + len(c_str) : ].strip()
    return text


def case_ino_mapping(data: str, cn: str) -> Tuple[str, str] :
    data = data.split("\n")
    data_0 = data[0].split(' ')
    out = ""
    out += f"Case number #{cn} created at {f'{data_0[-2].strip()} {data_0[-1].strip()}'}.\n "
    out += f"Service Level of case number #{cn} is {data[1].split(':')[1].strip()}.\n "
    out += f"Product of case number #{cn} is {data[2].split(':')[1].strip()}.\n"
    out += f"Specialization of case number #{cn} is {data[3].split(':')[1].strip()}.\n"
    out += f"IRT SLO met of case number #{cn} : {data[4].split(':')[1].strip()}.\n"
    out += f"TRT SLO met of case number #{cn} : {data[5].split(':')[1].strip()}.\n"
    out += f"Channel of case number #{cn} is {data[6].split(':')[1].strip()}.\n"
    out += f"Close Status of case number #{cn} is {data[7].split(':')[1].strip()}.\n"
    out += f"Subject of case number #{cn} is {data[8].split(':')[1].strip()}.\n"
    out += f"Shard of case number #{cn} is {data[9].split(':')[1].strip()}.\n"
    return out, f'{data_0[-2].strip()} {data_0[-1].strip()}'

def format_date(d) -> str:
    d3 = str(d).split(" ")
    d4 = d3[-1].split(":")
    if len(d3) > 1:
        return f"{d3[0]} {d3[1]} {d4[0]} hours {d4[1]} minutes {d4[2]} seconds"
    else:
        return f"{d4[0]} hours {d4[1]} minutes {d4[2]} seconds"


def SLA_tm(data: dict, ct_date: str) -> str:
    ct_date = ct_date.split(" ")[0] + " " + ct_date.split(" ")[1].split("-")[0]
    ct_date = datetime.strptime(ct_date, "%Y-%m-%d %H:%M:%S")
    cust = data["Customer"]
    supp = data["Support"]
    fmr_date = supp[0]
    out = f"Support sent FMR or first meaningful response message after {format_date(fmr_date - ct_date)} time.\n"
    counter = 0
    for i in cust[1:]:
        for j in supp[1:]:
            if j > i:
                counter += 1
                d1 = j - i
                out += f"Support sent INCRUP #{counter} after {format_date(d1)} time after Customer's reply.\n"
                break
    return out


def create_data(r: Any, file: Union[pd.DataFrame, str], output_mode: str = "string"):
    if isinstance(type(file), str):
        df_others = pd.read_csv(file)
    else:
        df_others = file
        df_others["time"] = df_others["timestamp"].apply(lambda x: datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))
    df_others["role"] = df_others["role"].fillna("System")
    inp_data_oth = []
    flag = False

    for i in range(len(df_others)):
        try:
            if (df_others.iloc[i]["type"] == "STATUS"):
                if (pd.isnull(df_others.iloc[i]["previous_value"])):
                    inp_data_oth.append("The case status of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " on " + df_others.iloc[i]["time"] + ". ")
                else:
                    inp_data_oth.append("The case status of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " from " + df_others.iloc[i]["previous_value"] + " on " + df_others.iloc[i]["time"] + ". ")
            elif (df_others.iloc[i]["type"] == "CASE_INFO"):
                inp_data_oth.append(case_ino_mapping(df_others.iloc[i]["new_value"], str(df_others.iloc[i]["case_number"]))[0])
                creation_date = case_ino_mapping(df_others.iloc[i]["new_value"], str(df_others.iloc[i]["case_number"]))[1]

            elif (df_others.iloc[i]["type"] == "PRIORITY_CHANGE"):
                if (pd.isnull(df_others.iloc[i]["previous_value"])):
                    inp_data_oth.append("The case priority of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " on " + df_others.iloc[i]["time"] + ". ")
                else:
                    inp_data_oth.append("The case priority of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " from " + df_others.iloc[i]["previous_value"] + " on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "OWNER_CHANGE"):
                if (pd.isnull(df_others.iloc[i]["previous_value"])):
                    inp_data_oth.append("The case owner of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " on " + df_others.iloc[i]["time"] + ". ")
                else:
                    inp_data_oth.append("The case owner of the case number #" + str(df_others.iloc[i]["case_number"]) + " was changed to " + df_others.iloc[i]["new_value"] + " from " + df_others.iloc[i]["previous_value"] + " on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "CASE_COMMENTS"):
                if ("Portal comment from customer" in df_others.iloc[i]["new_value"]):
                    continue
                inp_data_oth.append("On " + df_others.iloc[i]["time"] + " for case number #" + str(df_others.iloc[i]["case_number"]) + " : " + df_others.iloc[i]["new_value"] + ". ")

            elif (df_others.iloc[i]["type"] == "ESCALATION_CREATED"):
                inp_data_oth.append("Case number #"+ str(df_others.iloc[i]["case_number"]) + " escalated (details : " + df_others.iloc[i]["new_value"] + ") on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "ESCALATION_ENDED"):
                inp_data_oth.append("Escalation (details : " + df_others.iloc[i]["new_value"] + ") for case number #" + str(df_others.iloc[i]["case_number"]) + " closed or resolved on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "ESCALATION_DESCRIPTION"):
                inp_data_oth.append("Escalation reason and detailed description for case number #" + str(df_others.iloc[i]["case_number"]) + " : " + df_others.iloc[i]["new_value"] + ". ")

            elif (df_others.iloc[i]["type"] == "CONSULT"):
                inp_data_oth.append("Case consult (details : " + df_others.iloc[i]["new_value"] + ") for case number #" + str(df_others.iloc[i]["case_number"]) + " was created on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "BUG"):
                inp_data_oth.append("A bug (details : " + df_others.iloc[i]["new_value"] + ") related to the case number #" + str(df_others.iloc[i]["case_number"]) + " was created on " + df_others.iloc[i]["time"] + ". ")

            elif (df_others.iloc[i]["type"] == "FEEDBACK"):
                feed_back_score = f"Feedback for the case number #{df_others.iloc[i]['case_number']} : "
                rate = df_others.iloc[i]['new_value'][0]
                tag_body = "Email - "
                em_body_idx = df_others.iloc[i]['new_value'].find(tag_body)
                if (em_body_idx < 0):
                    if (int(rate) < 4):
                        feed_back_score += f"Support agent got DSAT (Rating - {rate})."
                    else:
                        feed_back_score += f"Support agent got CSAT (Rating - {rate})."
                else:
                    em_body = df_others.iloc[i]['new_value'][em_body_idx + len(tag_body) : ]
                    if (int(rate) < 4):
                        feed_back_score += f"Support agent got DSAT (Rating - {rate}).\n Customer's feedback (Reason for DSAT) : {em_body}"
                    else:
                        feed_back_score += f"Support agent got CSAT (Rating - {rate}).\n Customer's feedback (Reason for CSAT) : {em_body}"

                inp_data_oth.append(feed_back_score)

            elif (df_others.iloc[i]["type"] == "CASE_MESSAGES" and not flag and df_others.iloc[i]["role"] == "Support"):
                inp_data_oth.append("First meaningful response message or FMR or initial response of the case number #" + str(df_others.iloc[i]["case_number"]) + " to the Customer was sent by Support on " + df_others.iloc[i]["time"] + ".\n Below is the FMR message content:\n " + preProcess(df_others.iloc[i]["new_value"]).replace("\n"," ") + "\n")
                flag = True


        except Exception as e:
            print(e)
            continue

    if(output_mode == "string"):
        out_oth_str = f"A. Below are the whole case information, timeline, activity history of the case #{df_others.iloc[0]['case_number']} : \n\n"
        out_oth_str += break_chunks_v2(inp_data_oth)
        r.insert_docs([out_oth_str], "chunk")
    
        out_oth_str = f"\n\nB. Below are the whole conversation and reply messages between Customer and Support along with the time of the message sent of the case #{df_others.iloc[0]['case_number']} : \n\n"

    #df_msgs = df_others[df_others["type"] == "CASE_MESSAGES"]
    df_msgs = df_others[df_others["type"].isin(["CASE_MESSAGES", "CUSTOMER_CHAT"])]
    out_oth_str_msg = ""
    sender_map = {}
    for i in range(len(df_msgs)):
        if(pd.isnull(df_msgs.iloc[i]["new_value"])):
            continue
        if df_msgs.iloc[i]['role'] not in sender_map.keys():
            sender_map[df_msgs.iloc[i]['role']] = []
        sender_map[df_msgs.iloc[i]['role']].append(datetime.strptime(df_msgs.iloc[i]['time'], "%Y-%m-%d %H:%M:%S"))
        #out_oth_str_msg += "Message from " + df_msgs.iloc[i]['role'] + " on " + df_msgs.iloc[i]['time'] + " : " + preProcess(df_msgs.iloc[i]['new_value']).replace("\n"," ") + ".\n\n"
        if df_msgs.iloc[i]['type'] == "CUSTOMER_CHAT":
            out_oth_str_msg += "Chat response from " + df_msgs.iloc[i]['role'] + " on " + df_msgs.iloc[i]['time'] + " : " + preprocess_block_chat_transcript(df_msgs.iloc[i]['new_value']) + ".\n\n"
        else:
            out_oth_str_msg += "Message from " + df_msgs.iloc[i]['role'] + " on " + df_msgs.iloc[i]['time'] + " : " + preProcess(df_msgs.iloc[i]['new_value']).replace("\n"," ") + ".\n\n"


    r.insert_docs([out_oth_str + out_oth_str_msg], "chunk")

    out_oth_str_msg = f"\n\nC. Below are the total messages counts and SLA details of FMR and INCRUP messages sent by support of the case #{df_others.iloc[0]['case_number']} : \n\n"


    out_oth_str = out_oth_str_msg + f"Total number of messages sent by Customer is : {len(sender_map['Customer'])} .\nTotal number of messages sent by Support is : {len(sender_map['Support'])} .\n" + SLA_tm(sender_map, creation_date)

    r.insert_docs([out_oth_str], "chunk")


    df_msgs = df_others[df_others["type"] == "SME_CHAT"]
    if len(df_msgs) > 0:
        out_oth_str_msg = ""
        out_oth_str = f"\n\nD. Below are the chat conversation between SME (Subject matter expert) and the Support agent or TSR of the case #{df_others.iloc[0]['case_number']} : \n\n"
        for i in range(len(df_msgs)):
            if(pd.isnull(df_msgs.iloc[i]["new_value"])):
                continue
            #out_oth_str_msg += " On " + df_msgs.iloc[i]['time'] + " message from " + df_msgs.iloc[i]['new_value'].replace("\n"," ") + ".\n\n"
            out_oth_str_msg += "Chat response from " + df_msgs.iloc[i]['role'] + " on " + df_msgs.iloc[i]['time'] + " : " + preprocess_block_chat_transcript(df_msgs.iloc[i]['new_value']) + ".\n\n"

        r.insert([out_oth_str + out_oth_str_msg], "chunk")




    
