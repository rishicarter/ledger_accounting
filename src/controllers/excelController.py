import pandas as pd
import numpy as np
import os

import src.services.documentsModel as docModel
import src.services.embeddingsModel as emModel

def preprocess(df: pd.DataFrame):
    df = df.copy()
    index = df[df.iloc[:,0] == 'Date'].index[0]
    df1 = df.iloc[index+1:,]
    current_cols = df1.columns
    modified_cols = ['date', 'direction', 'transaction_desc', 'vch_type', 'vch_no', 'debit', 'credit']
    df1 = df1.rename(columns=dict(zip(current_cols, modified_cols)))
    df1['transaction_details'] = df1.apply(lambda row: row['transaction_desc'] if pd.isna(row['date']) and pd.isna(row['direction']) else None, axis=1)
    df1['transaction_details'] = df1['transaction_details'].bfill()
    df1 = df1[~df1['transaction_desc'].str.contains('Opening Balance|Closing Balance', case=False, na=False)]
    df1 = df1.dropna(subset=['date', 'direction', 'transaction_desc', 'vch_type', 'vch_no'])
    df1['direction'] = df1['direction'].apply(lambda x: "No" if x == "By" else "Yes")
    df1['debit'] = df1['debit'].fillna(df1['credit'])
    df1.rename(columns={"debit": "amount", "direction": "is_amount_debited"}, inplace=True)
    df1.drop('credit', axis=1, inplace=True)
    
    return df1


def load_df(dfs_list):
    dfs = []
    for df in dfs_list:
        df = preprocess(df)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    # df.to_csv(OUTPUT_PATH+OUT_FILENAME, index=False)
    return df


def handle_uploaded_file(uploaded_file, save_to_db=False):
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names
    dfs = []
    for sheet_name in sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        dfs.append(df)
    
    ledger_df = load_df(dfs)
    if save_to_db:
        docModel.save_documents(ledger_df.to_dict("records"))

    return ledger_df
