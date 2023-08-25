# Get Project Metrics

### Pypi Downloads
The pypi dataset is publicly available in BigQuery. When you create a google cloud account you'll get more than enough credits to make the BigQuery queries to get project usage. 

Get BigQuery Access Token:
- Create google account
- Create a service account with BigQuery access
- Generate a key for the service account for the `pypi-packages-query-key.json` file


### GitHub Stars
Here you'll need to access the GitHub API. The rate limits for un-authenticated accounts are pretty low, so you'll want to create a GitHub API key 

Get Github Access Token:
- Go to your github account
- Go to Settings then Developer Settings
- Create a fine grained personal access token
- Add your personal access token to the `github.config` file


### Files
```bash
get_metrics.py                       <-- main python script
github.config.exmple                 <-- example GH config file (remove .example extension in your code)
pypi-package-query-key.json.exmple   <-- example BQ file (remove .example extension in your code)
requirements.txt                     <-- python requirments
charts/                              <-- folder to save figures to
```



### Run the script
(after updating the `github.config` and `pypi-package-query-key.json` files)
```bash
pip install -r requirement.txt 
python get_metrics.py
```