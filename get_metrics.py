from google.cloud import bigquery
import os
import pandas as pd

import requests
import json
import time
from dataclasses import dataclass
import configparser

import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level = logging.INFO)

@dataclass
class Package:
    pypi_name: str
    github_name: str
    github_owner: str
    
    # optional tags
    from_snowflake: bool = False
    db_connector: bool = False
    ds_tool: bool = False
    new_project: bool = False
    pipelining: bool = False
    
    # not an easy way to get in bulk, so just mark this
    stars_10K_plus: bool = False



# ----------  Pypi Downloads  -------------
    
def get_pypi_data(package:Package, client:bigquery.Client, months_history:int = 36) -> bigquery.job.query.QueryJob:

    logging.info(f"getting bigquery data for {package.pypi_name}")

    query_downloads = f"""
        SELECT COUNT(*) AS num_downloads,
               TIMESTAMP_TRUNC(timestamp, WEEK) as week
            FROM `bigquery-public-data.pypi.file_downloads`


        WHERE file.project = '{package.pypi_name}'
        AND DATE(timestamp)
            BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {months_history} MONTH)
            AND CURRENT_DATE() 
        GROUP BY week
    """
    
    query_job = client.query(query_downloads)
        
    return query_job



def bq_to_pandas(query_job_id: bigquery.job.query.QueryJob, package:str) -> pd.DataFrame:
    
    results = query_job_id.result()

    row_results = []
    for row in results:
        row_results.append([row.num_downloads, row.week])
        
    df = pd.DataFrame(row_results, columns = [f'{package}_downloads', 'week_date'])
    
    df = df.sort_values(by="week_date")
    
    return df
    

    

def process_downloads(package_list:list, client:bigquery.Client , query_func:callable, process_query: callable) -> pd.DataFrame:

    logging.info(f"processing biquery data....")

    query_ids = []
    for package in package_list:
        query_id = query_func(package, client)
        query_ids.append(query_id)
        
    dfs = []
    for i, query_id in enumerate(query_ids):
        df = process_query(query_id, package_list[i].pypi_name)
        dfs.append(df)
        
    # merge queries  
    dfs = [df.set_index('week_date') for df in dfs]
    
    packages_df = dfs[0].join(dfs[1:])
        
    return packages_df





# ---------   Github Stars   ----------

def get_github_results(package:Package, gh_token:str) -> list:
    # we'll get a max of 10k stars, otherwise just mark as over 10000
    logging.info(f"getting github api data for {package.github_name}")
    
    star_history = []
    max_pages = 100
    page_results = [1]
    page_num = 0
    
    while len(page_results) != 0 and page_num < max_pages:

        url = f"https://api.github.com/repos/{package.github_owner}/{package.github_name}/stargazers?per_page=100&page={page_num}"
        repo_response = requests.get(url,headers={'Accept': 'application/vnd.github.v3.star+json', "Authorization": f"Bearer {gh_token}"})
        page_results = json.loads(repo_response.text)

        star_history.extend(page_results)
        page_num += 1
        # rate limit pages
        time.sleep(0.05)

    # rate limit between packages
    time.sleep(1)

    
        
    return star_history
     
    
def github_star_aggregation(star_history: list, package:Package) -> pd.DataFrame:
    

    star_hist_df = pd.DataFrame(star_history)
    
    logging.debug(star_history)
    
    # really we just need the counts and date here
    star_hist_df['starred_at'] = pd.to_datetime(star_hist_df['starred_at'])
    star_hist_df['count'] = 1
    
    star_hist_df['month'] = star_hist_df['starred_at'].dt.to_period("M").dt.to_timestamp()
    stars_per_month_df = star_hist_df.groupby(['month']).sum().reset_index()
    
    stars_per_month_df = stars_per_month_df.rename(columns={"count": f"{package.github_name}_stars"})
    
    
    return stars_per_month_df




def process_github_stars(package_list:list, query_func:callable, process_query: callable, gh_token:str):
    
    logging.info(f"processing github data....")

    dfs = []
    for package in package_list:
        if not package.stars_10K_plus:
            star_history = get_github_results(package, gh_token)

            stars_per_month_df = github_star_aggregation(star_history, package)
            dfs.append(stars_per_month_df)
            
        
    dfs = [df.set_index('month') for df in dfs]
    
    packages_df = dfs[0].join(dfs[1:])
    
    return packages_df
    

def downloads_chart(packages: list, packages_df:pd.DataFrame, tag:str)-> None:
    sns.set_theme()
    sns.set(font_scale=2) 
    plt.xticks(rotation=45)
    sns.set(rc={'figure.figsize':(20,12)})
    plt.ylabel("Downloads")
    plt.xlabel("Date")
    #plt.title("Snowflake Project Downloads Over Time")
    
    package_names = []
    for package in packages:
        if getattr(package, tag):
            package_names.append(f"{package.pypi_name}_downloads")
            
    plot = sns.lineplot(
                data = packages_df[package_names],
                 markers=True,
                legend="full")
    
    plt.legend(fontsize='18')
    fig = plot.get_figure()
    fig.savefig(f"charts/downloads_{tag}.png", bbox_inches='tight') 
    plt.close()
    
    return 

    
            
def stars_chart(packages: list, stars_df: pd.DataFrame, tag:str) -> None:
    sns.set_theme()
    sns.set(font_scale=2) 
    plt.xticks(rotation=45)
    sns.set(rc={'figure.figsize':(20,12)})
    plt.ylabel("Stars")
    plt.xlabel("Date")    
    
    package_names = []
    for package in packages:
        if package.stars_10K_plus:
            continue 
        
        if tag == "all" and not package.stars_10K_plus:
            package_names.append(f"{package.github_name}_stars")
        
        else:
            if getattr(package, tag) and not package.stars_10K_plus:
                package_names.append(f"{package.github_name}_stars")
            
    plot = sns.lineplot(
                data = stars_df[package_names],
                 markers=True,
                legend="full")
    
    plt.legend(fontsize='18')
    fig = plot.get_figure()
    fig.savefig(f"charts/stars_{tag}.png", bbox_inches='tight') 
    plt.close()
    
    return







if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read("github.config")
    gh_token = config['default']['token']

    # create service account with biquery permissions to get .json file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'pypi-packages-query-key.json'
    # this will use the env variable
    client = bigquery.Client()



    # projects to examine:
    packages = [
        Package("snowflake-connector-python", "snowflake-connector-python", "snowflakedb", from_snowflake=True, db_connector=True),
        Package("snowflake-snowpark-python", "snowpark-python", "snowflakedb", from_snowflake=True, ds_tool=True, new_project=True),
        Package("snowflake-ml-python", "snowflake-ml-python", "snowflakedb", from_snowflake=True, ds_tool=True, new_project=True),
        Package("SQLAlchemy", "sqlalchemy", "sqlalchemy", db_connector=True),
        Package("cx-Oracle", "python-cx_Oracle", "oracle", db_connector=True),
        Package("psycopg2", "psycopg2", "psycopg", db_connector=True),
        
        Package("scikit-learn", "scikit-learn", "scikit-learn",  ds_tool=True, stars_10K_plus=True),
        Package("pyspark", "spark", "apache", ds_tool=True, stars_10K_plus=True),
        Package("tensorflow", "tensorflow", "tensorflow", ds_tool=True, stars_10K_plus=True),
        Package("transformers", "transformers", "huggingface", ds_tool=True, stars_10K_plus=True),
        Package("jax", "jax", "google", ds_tool=True, stars_10K_plus=True),
        Package("torch", "pytorch", "pytorch", ds_tool=True, stars_10K_plus=True),
        Package("sagemaker", "sagemaker-python-sdk", "aws",  ds_tool=True, pipelining=True),
        
        Package("kedro", "kedro", "kedro-org", ds_tool=True, new_project=True, pipelining=True),
        Package("zenml", "zenml", "zenml-io", ds_tool=True, new_project=True),
        Package("dagster", "dagster", "dagster-io"),
        Package("prefect", "prefect", "PrefectHQ", stars_10K_plus=True, pipelining=True),
        Package("duckdb", "duckdb", "duckdb", stars_10K_plus=True),
        Package("snowflake-ice-pick", "ice_pick", "PrestonBlackburn", new_project=True),
        Package("apache-airflow", "airflow", "apache", stars_10K_plus=True, pipelining=True),
        Package("pandas", "pandas", "dev-pandas",  ds_tool=True, stars_10K_plus=True),
        Package("polars", "polars", "pola-rs", ds_tool=True, stars_10K_plus=True, new_project=True)
    ]  


    # get data:
    packages_df = process_downloads(packages, client, get_pypi_data, bq_to_pandas)
    packages_df.to_csv('package_download_history.csv')
    # packages_df = pd.read_csv('package_download_history.csv')


    stars_df = process_github_stars(packages, get_github_results, process_github_stars, gh_token)
    stars_df.to_csv('star_history.csv')
    # stars_df = pd.read_csv('star_history.csv')



    # generate charts:
    downloads_chart(packages, packages_df, 'from_snowflake')
    downloads_chart(packages, packages_df, 'ds_tool')
    downloads_chart(packages, packages_df, 'new_project')
    downloads_chart(packages, packages_df, 'db_connector')
    downloads_chart(packages, packages_df, 'pipelining')


    stars_chart(packages, stars_df, 'from_snowflake')
    stars_chart(packages, stars_df, 'ds_tool')
    stars_chart(packages, stars_df, 'new_project')
    stars_chart(packages, stars_df, 'db_connector')
    stars_chart(packages, stars_df, 'pipelining')
    stars_chart(packages, stars_df, 'all')








