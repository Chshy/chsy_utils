import subprocess
import requests
import json
import time
from datetime import datetime, timedelta
import sys

def send_pushplus_message(token, title, content):
    url = "http://www.pushplus.plus/send"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "token": token,
        "title": title,
        "content": content
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def is_work_time():
    now = datetime.now()
    # 定义工作时间段 (例如 9:00 到 22:00)
    start_work_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end_work_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
    # 检查当前时间是否在工作时间内
    return start_work_time <= now <= end_work_time

def wait_until_work_time():
    now = datetime.now()
    start_work_time = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    time_to_wait = (start_work_time - now).total_seconds()
    time.sleep(time_to_wait)

# 你的PushPlus token
token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" #
title = "训练完成通知"

# 从命令行参数获取训练脚本和其参数
if len(sys.argv) < 2:
    print("请提供训练脚本文件名和参数")
    sys.exit(1)

train_script = sys.argv[1]
train_script_args = sys.argv[2:]

# 启动并监控训练脚本
process = subprocess.Popen(["python", train_script] + train_script_args)

# 定时检查训练脚本的状态
while process.poll() is None:
    time.sleep(60)  # 每60秒检查一次

# 获取训练完成时间
completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
content = f"你的训练程序 {train_script} 已于 {completion_time} 完成。"

# 打印完成信息
print('=' * 25)
print(f"[{completion_time}] 程序已完成.")

# 训练脚本完成后，发送通知
if is_work_time():
    send_pushplus_message(token, title, content)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 微信通知已发送.")
else:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 不是工作时间.等待工作时间发送通知.")
    wait_until_work_time()
    send_pushplus_message(token, title, content)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 微信通知已发送.")

