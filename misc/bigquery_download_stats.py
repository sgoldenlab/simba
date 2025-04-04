from google.cloud import bigquery
import pandas as pd
import os

# Write credentials to file from GitHub secret
with open("gcp-key.json", "w") as f:
    f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"])

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"

# Initialize BigQuery client
client = bigquery.Client()

# SQL query
query = """
SELECT
  file.project AS package_name,
  file.version AS package_version,
  country_code AS country,
  DATE(timestamp) AS download_date,
  COUNT(*) AS download_count
FROM `bigquery-public-data.pypi.file_downloads`
WHERE file.project = 'simba-uw-tf-dev'
  AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY package_name, package_version, country, download_date
ORDER BY download_date DESC, package_version, country
"""

query_job = client.query(query)
rows = query_job.result()

# Convert and save CSV
df = pd.DataFrame([dict(row) for row in rows])
df.to_csv("misc/bigquery_download_stats.csv", index=False)
