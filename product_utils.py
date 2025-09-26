import numpy as np
import pandas as pd

product_df = pd.read_csv('D:/datasets/Product Health/eda_product.csv', index_col=0)

def recency_score(row, ref_date):
    days_since_sale = (ref_date - row["Recent_Sale"]).days
    total_days = row["Days_in_Market"]

    if days_since_sale <= 31:
        recency_component = 1 - np.log1p(days_since_sale / total_days)
    else:
        recency_component = 1 - ((days_since_sale / total_days))
    recency_component = max(0, min(recency_component, 1))
    return row["Popularity_Score"] * recency_component

def lookup_details(prod_id):
    return product_df[product_df["Product_ID"]==prod_id]

def prod_pred(row, ref_date):
    row["Recent_Sale"] = pd.to_datetime(row["Recent_Sale"])
    row["Popularity_Score"] = (row["Unique_purchase_dates"] / row["Days_in_Market"])
    row['Relevance_Score'] = recency_score(row, ref_date)
    cluster_summary = product_df.groupby("Product_Cluster")[["Popularity_Score", "Relevance_Score"]].mean()
    if row['Popularity_Score']+row['Relevance_Score']<=cluster_summary.loc[0,'Popularity_Score']+cluster_summary.loc[0,'Relevance_Score']:
        row["Product_Cluster"] = 0
    elif row['Popularity_Score']+row['Relevance_Score']<=cluster_summary.loc[1,'Popularity_Score']+cluster_summary.loc[1,'Relevance_Score']:
        row["Product_Cluster"] = 1
    else:
        row['Product_Cluster'] = 2
    return row
