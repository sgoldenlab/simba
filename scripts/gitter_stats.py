import json
import pandas as pd
import re
from datetime import datetime

DATA_PATH = r"C:\Users\sroni\Downloads\gitter_chat.json"
SAVE_PATH = r'C:\Users\sroni\OneDrive\Desktop\gitter_chat_stats.json'

with open(r"C:\Users\sroni\Downloads\gitter_chat.json", "r", encoding='utf-8') as f:
    data = json.load(f)

results = pd.DataFrame(columns=['USERNAME', 'BODY'])
for message in data['messages']:
    msg_sender = message['sender']
    msg_sender = msg_sender.replace('@', ' ')
    msg_sender = re.sub(r'\W+', '', msg_sender)
    if 'body' in message['content'].keys():
        body = message['content']['body']
        body = body.replace('\n', ' ')
        body = body.replace('\t', ' ')
    else:
        body = ' '
    body = re.sub(r'\W+', ' ', body)
    results.loc[len(results)] = [msg_sender, body]

post_cnt = len(results)
unique_posters = results['USERNAME'].nunique()
average_user_post_cnt = results['USERNAME'].value_counts().mean()

results = {'post_cnt': len(results), 'unique_users_with_posts': unique_posters, 'avg_posts_per_user': average_user_post_cnt, 'date': datetime.today().strftime('%Y-%m-%d')}


with open(SAVE_PATH, 'w') as f:
    json.dump(results, f, indent=4)
#results.to_csv(SAVE_PATH, index=None)


