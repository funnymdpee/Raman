import time
import psutil
import subprocess

# 用来检查某个进程是否在运行的函数
def is_running(script_name):
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.info['cmdline'] and script_name in " ".join(p.info['cmdline']):
            return True
    return False

# 等待训练脚本 a.py 结束
print("等待训练任务结束...")
while is_running("train.py"):  # 这里可以替换成你的训练脚本的名字
    print("last exe is running!")
    time.sleep(60)  # 每10秒检查一次

print("train_backup.py 训练完成，开始执行 train.py")
subprocess.run(["python", "train.py"])
#
# # 等待 b.py 结束后再执行 c.py
# while is_running("b.py"):
#     time.sleep(10)  # 等待 b.py 结束
#
# print("b.py 训练完成，开始执行 c.py")
# subprocess.run(["python", "c.py"])
