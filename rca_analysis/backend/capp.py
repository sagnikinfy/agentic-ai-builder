import os
import random
import time
from flask import Flask, request, render_template, session, flash, redirect, url_for, jsonify
from celery import Celery
from helper import *
from main import RAG
from main_1 import RAG as QRAG, QueryParam
import json
import redis
from datetime import datetime
import re
from load_data import *

password = ""
user = ""


app = Flask(__name__)
app.config['SECRET_KEY'] = ''
app.config['CELERY_BROKER_URL'] = f'redis://{user}:{password}@localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = f'redis://{user}:{password}@localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@celery.task(bind=True)
def insert_task(self, cn, em):
    total = ', '.join(cn)
    counter = 0
    tot_completed = ""
    error_logs = ""
    for i in cn:
        try:
            if not check_if_exists(i):
                if counter == 0:
                    self.update_state(state='RUNNING', meta={'current': f"case #{i} running", 'total': f"out of {total} cases", 'status': f"0 cases completed", "submitted_by" : em})
                else:
                    self.update_state(state='RUNNING', meta={'current': f"case #{i} running", 'total': f"out of {total} cases", 'status': f"{tot_completed} cases completed", "submitted_by" : em})
                dir_name = "adv_rag"
                dir_path = f"/home/sagnikr/backend/{dir_name}/{i}"
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                r = RAG(working_dir = f"{dir_name}/{i}")
                df = extract_data_from_cn(int(i))
                create_data(r, df)
                upload_local_directory_to_gcs(f"{dir_name}/{i}", f"adv_rag/{i}")
                if check_if_exists(i):
                    delete_local_folders(dir_name, i)
                    if counter == 0:
                        self.update_state(state='RUNNING', meta={'current': f"case #{i} completed", 'total': f"out of {total} cases", 'status': f"{i} case completed", "submitted_by" : em})
                    else:
                        self.update_state(state='RUNNING', meta={'current': f"case #{i} completed", 'total': f"out of {total} cases", 'status': f"{tot_completed} case completed", "submitted_by" : em})
                        
            else:
                self.update_state(state='RUNNING', meta={'current': f"case #{i} skipped", 'total': f"out of {total} cases", 'status': f"case #{i} already there, skipping..", "submitted_by" : em})

            counter += 1
            tot_completed = ', '.join(cn[:counter])
                
        except Exception as e:
            error_logs += f"while insering case #{i} exception occurred : {str(e)}\n"
            continue

    log_data = {'current': f"{tot_completed} case completed", 'total': f"out of {total} cases", 'status': 'Task completed, log data inserted!', 'result' : f"{counter}/{len(cn)} cases inserted.. completed..", "errors" : error_logs, "submitted_by" : em}
    
    is_insert = insert_log(em, log_data, datetime.now())

    if not is_insert:
        log_data['result'] = 'Task completed, log data failed to insert due to unexpected error'
        

    return log_data
            
            
        

@app.route("/genrca", methods=["POST"])
def gen_rca():
    data = request.json
    cn = data["cn"]
    prompt = data["prompt"]
    try:
        r = QRAG(storage_path = f"adv_rag/{cn}")
        history = ""
        mode = "hybrid"
        out = r.query_async(prompt, history, QueryParam(mode = mode, only_need_prompt=False, only_need_context=False))
        return jsonify({
            "predictions": out
        })
    except Exception as e:
        return jsonify({
            "predictions": str(e)
        })
        
    
@app.route('/loadtask', methods=['POST'])
def longtask():
    data = request.json
    cns = list(map(str.strip, data["cn"].split(",")))
    em = data["em"]
    task = insert_task.apply_async(args = (cns, em))
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route("/get_jobs", methods=['GET'])
def get_all_running_job():
    try:
        out_data= []
        r = redis.from_url(f'redis://{user}:{password}@localhost:6379/0')
        for s in r.scan_iter("celery-task-meta*"):
            out = json.loads(r.get(s).decode('UTF-8'))
            if(out['status'] == "RUNNING"):
                out_data.append({"JobID" : f"/status/{out['task_id']}", "Submitted By" : out['result']['submitted_by']})
        return jsonify({
            "predictions": out_data
        })
    except Exception as e:
        return jsonify({
            "predictions": str(e)
        })



@app.route("/get_available_cases", methods = ["POST"])
def get_all_available_cases():
    try:
        submitted_cns = []
        cn_arr = request.json["cn"]
        # layer 1 check at bucket level
        not_available = check_if_exists_case_arr(cn_arr)
        if not not_available:
            return jsonify({
                "predictions": [[],[],[]]
            })
        else:
            # layer 2 check at job level
            r = redis.from_url(f'redis://{user}:{password}@localhost:6379/0')
            for s in r.scan_iter("celery-task-meta*"):
                out = json.loads(r.get(s).decode('UTF-8'))
                if(out['status'] == "RUNNING"):
                    submitted_cns.extend(re.findall(r'\d+(?:,\d+)?', out['result']['total']))
            #print(submitted_cns)
            not_available_in_jobs = list(set(not_available) - set(submitted_cns))
            #print(not_available_in_jobs)
            in_bucket = list(set(cn_arr) - set(not_available))
            if not not_available_in_jobs:
                return jsonify({
                    "predictions": [[],not_available,in_bucket]
                })
            else:
                # strictly available cases not in queue or bucket
                queued = list(set(not_available) - set(not_available_in_jobs))
                #print(queued)
                return jsonify({
                    "predictions": [not_available_in_jobs,queued,in_bucket]
                })
        
    except Exception as e:
        return jsonify({
            "predictions": str(e)
        })
            
        


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = insert_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': "calculating..",
            'status': 'Pending...',
            "submitted_by" : "checking..."
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            "submitted_by" : task.info.get("submitted_by", "")
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
            response['errors'] = task.info['errors']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info)
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)



