import os
import time
import logging
import smtplib
from email.mime.text import MIMEText

COOLDOWN_SECONDS = 300
last_email_time = 0

def send_email(subject, body, client_email):
    global last_email_time
    current_time = time.time()
    if current_time - last_email_time < COOLDOWN_SECONDS:
        remaining_time = int(COOLDOWN_SECONDS - (current_time - last_email_time))
        logging.warning(f"郵件發送過於頻繁，請在 {remaining_time} 秒後重試！")
        return False

    server_email = "huanghongjunh2@gmail.com"
    password = "kcco ywii rcdj cgem"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = server_email
    msg['To'] = client_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(server_email, password)
            server.sendmail(server_email, client_email, msg.as_string())
            logging.info("郵件發送成功！")
            last_email_time = time.time()
            return True
    except Exception as e:
        logging.error(f"郵件發送失敗：{str(e)}")
        return False