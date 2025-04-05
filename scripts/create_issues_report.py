import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import os
from zipfile import ZipFile

# === CONFIG ===
REPO = "sgoldenlab/simba"  # Repo: sgoldenlab/simba
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Get the GitHub token from environment variables
PER_PAGE = 100
HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# === PAGINATION HANDLER ===
def fetch_paginated(url, params=None):
    results = []
    page = 1
    while True:
        query = params.copy() if params else {}
        query.update({'per_page': PER_PAGE, 'page': page})
        response = requests.get(url, headers=HEADERS, params=query)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
        page_data = response.json()
        if not page_data:
            break
        results.extend(page_data)
        page += 1
        time.sleep(0.1)
    return results

# === FETCH ISSUES ===
def fetch_all_issues():
    print("Fetching issues...")
    return fetch_paginated(f"https://api.github.com/repos/{REPO}/issues", {"state": "all"})

# === FETCH COMMENTER USERNAMES AND LAST COMMENT DATE ===
def get_comment_usernames_and_last_comment_date(issue_number):
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    try:
        comments = fetch_paginated(url)
        usernames = {c["user"]["login"] for c in comments}
        last_comment_date = max(c["created_at"] for c in comments) if comments else None
        return sorted(usernames), last_comment_date
    except Exception as e:
        print(f"Failed on issue #{issue_number}: {e}")
        return [], None

# === MAIN FUNCTION ===
def get_issues_dataframe():
    issues = fetch_all_issues()
    print(f"Fetched {len(issues)} total issues.")

    # Remove pull requests
    issues = [i for i in issues if "pull_request" not in i]
    print(f"Filtered to {len(issues)} issues (excluding PRs).")

    # Fetch comment usernames and last comment date in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        usernames_and_dates = list(executor.map(
            lambda i: get_comment_usernames_and_last_comment_date(i["number"]),
            issues
        ))

    data = []
    for issue, (commenters, last_comment_date) in zip(issues, usernames_and_dates):
        author = issue["user"]["login"]
        all_participants = set(commenters)
        all_participants.add(author)

        data.append({
            "issue_id": issue["number"],
            "author": author,
            "title": issue["title"],
            "commenter_usernames": sorted(commenters),
            "comment_count": issue.get("comments", 0),
            "unique_commenter_count": len(all_participants),
            "link": f"https://github.com/{REPO}/issues/{issue['number']}",
            "issue_post_date": issue["created_at"],
            "last_comment_date": last_comment_date,
        })

    df = pd.DataFrame(data)

    # Save DataFrame to CSV
    df.to_csv("/tmp/github_issues_report.csv", index=False)

    # Zip the CSV file
    with ZipFile("/tmp/github_issues_report.zip", 'w') as zipf:
        zipf.write("/tmp/github_issues_report.csv", os.path.basename("/tmp/github_issues_report.csv"))

    print("CSV and Zip file generated successfully!")

if __name__ == "__main__":
    get_issues_dataframe()
