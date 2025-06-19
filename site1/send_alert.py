import requests

# Dán webhook của bạn vào đây hoặc sử dụng webhook mẫu
WEBHOOK_URL = "https://discord.com/api/webhooks/1383805561841127517/jBFvE6RuGKaVU6ndR3BrHHtyTiqEYcsyIz7zGyIt62Imf3g6KfG1ebCSvD7VS5wNhSdr" 

def send_alert(message):
    payload = {"content": message}
    try:
        r = requests.post(WEBHOOK_URL, json=payload)
        if r.status_code in [200, 204]:
            print("[INFO] Alert sent to Discord.")
        else:
            print(f"[ERROR] Webhook failed with status code {r.status_code}")
    except Exception as e:
        print(f"[ERROR] Webhook exception: {e}")
