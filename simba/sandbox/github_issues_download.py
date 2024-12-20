import requests
import json

# Replace with your GitHub username, repository name, and personal access token
GITHUB_USERNAME = 'sgoldenlab'
GITHUB_REPOSITORY = 'simba'
GITHUB_TOKEN = ''

# GitHub API endpoint for issues
api_url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPOSITORY}/issues'

# Headers for authentication
headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_all_issues(api_url, headers):
    issues = []
    page = 1
    while True:
        response = requests.get(api_url, headers=headers, params={'state': 'all', 'page': page, 'per_page': 100})
        if response.status_code != 200:
            break
        page_issues = response.json()
        if not page_issues:
            break
        issues.extend(page_issues)
        page += 1
    return issues

# Get all issues
all_issues = get_all_issues(api_url, headers)

def get_comments(issue_number, headers):
    comments_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPOSITORY}/issues/{issue_number}/comments"
    response = requests.get(comments_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return []

def add_comments_to_issues(issues, headers):
    for issue in issues:
        issue_number = issue['number']
        comments = get_comments(issue_number, headers)
        issue['comments_data'] = comments
    return issues

# Add comments to issues
all_issues_with_comments = add_comments_to_issues(all_issues, headers)


def preprocess_issues_with_metadata(issues):
    processed_issues = []
    for issue in issues:
        print()
        processed_issue = {
            'title': issue.get('title', ''),
            'body': issue.get('body', ''),
            'user': issue['user'].get('login', ''),
            'reaction_cnt': issue['reactions'].get('total_count', ''),
            'created_at': issue.get('created_at', ''),
            'updated_at': issue.get('updated_at', ''),
            'author': issue.get('user', {}).get('login', ''),
            'comments': [{'body': comment['body'], 'created_at': comment['created_at'], 'author': comment['user']['login']} for comment in issue.get('comments_data', [])]
        }
        processed_issues.append(processed_issue)
    return processed_issues

# Preprocess issues with metadata
processed_issues_with_metadata = preprocess_issues_with_metadata(all_issues_with_comments)


def convert_issues_with_metadata_to_text(issues):
    text_data = ''
    for issue in issues:
        text_data += f"Title: {issue['title']}\n"
        text_data += f"Body: {issue['body']}\n"
        text_data += f"Created at: {issue['created_at']}\n"
        text_data += f"Updated at: {issue['updated_at']}\n"
        text_data += f"Author: {issue['author']}\n"
        text_data += f"Reactions: {issue['reaction_cnt']}\n"
        for comment in issue['comments']:
            text_data += f"Comment: {comment['body']}\n"
            text_data += f"Comment created at: {comment['created_at']}\n"
            text_data += f"Comment author: {comment['author']}\n"
        text_data += '\n' + '='*50 + '\n'
    return text_data

# Convert issues with metadata to text
issues_text_with_metadata = convert_issues_with_metadata_to_text(processed_issues_with_metadata)


with open('/Users/simon/Desktop/envs/simba/simba/simba/sandbox/github_issues.json', 'w', encoding='utf-8') as f:
    json.dump(issues_text_with_metadata, f, ensure_ascii=False, indent=4)


# Load data
with open('/Users/simon/Desktop/envs/simba/simba/simba/sandbox/github_issues.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
