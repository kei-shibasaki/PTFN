import subprocess
import json
import requests
import pandas as pd
import datetime
import time

JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')

DEFAULT_ATTRIBUTES = (
    'index',
    #'memory.total',
    'memory.free',
    'memory.used',
    #'name',
    #'uuid',
    #'timestamp',
    #'temperature.gpu',
    #'utilization.gpu',
    #'utilization.memory'
)

def gpuinfo2csv(out_path, nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    txt = 'datetime,server'
    for key in keys:
        txt = txt + f',{key}'
    txt = txt + '\n'

    with open(out_path, mode='w', encoding='utf-8') as fp:
        fp.write(txt)

    for line in lines:
        line = line.replace(', ', ',')
        with open(out_path, mode='a', encoding='utf-8') as fp:
            fp.write(f'{TIMESTAMP},{SERVER_INDEX},{line}\n')

def nortify_gpu_status(csv_path, omit_gpus=[], token_path='line_nortify_token.json'):
    def send_line_notify(line_notify_token, nortification_message):
        line_notify_api = 'https://notify-api.line.me/api/notify'
        headers = {'Authorization': f'Bearer {line_notify_token}'}
        data = {'message': f'{nortification_message}'}
        requests.post(line_notify_api, headers=headers, data=data)
    df = pd.read_csv(csv_path)

    for omit_gpu in omit_gpus:
        df = df[df['index']!=omit_gpu]
    free_gpu_info = df[df['memory.free']>15000]
    # print(f'{free_gpu_info}')
    if len(free_gpu_info)>0:
        with open(token_path, 'r', encoding='utf-8') as fp:
            token = json.load(fp)['token']
        send_line_notify(token, free_gpu_info)


if __name__=='__main__':
    SERVER_INDEX = 12
    csv_path = 'temp.csv'
    omit_gpus = [0,1]
    interval_minute = 20
    while True:
        TIMESTAMP = datetime.datetime.now(JST)
        TIMESTAMP = f'{TIMESTAMP.date()} {TIMESTAMP.hour:02}-{TIMESTAMP.minute:02}-{TIMESTAMP.second:02}'
        gpuinfo2csv(out_path=csv_path)
        nortify_gpu_status(csv_path, omit_gpus)
        time.sleep(interval_minute*60)