import schedule
import time
import os

def run_agent():
    os.system("python agent.py")

# poniedziałek 08:00
schedule.every().monday.at("08:00").do(run_agent)

# czwartek 08:00
schedule.every().thursday.at("08:00").do(run_agent)

while True:
    schedule.run_pending()
    time.sleep(60)
