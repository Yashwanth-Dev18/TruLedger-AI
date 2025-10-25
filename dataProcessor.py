import pandas as pd
import numpy as np
import os
import shutil

def process_transaction_data(input_file_path):
    """Process transaction data and save as processed CSV (Pandas version)"""

    try:
        # Load dataset
        print(f"📂 Loading dataset: {input_file_path}")
        df = pd.read_csv(input_file_path)

        # =============================================
        # 🔍 FEATURE ENGINEERING & DATA PREPROCESSING
        # =============================================

        # DROP UNNECESSARY COLUMNS
        drop_cols = [
            "index", "merchant", "first", "last", "gender", "street", "city", "zip",
            "city_pop", "trans_num", "unix_time"
        ]
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        # EXTRACT HOUR FROM 'trans_date_trans_time'
        df["txn_time"] = pd.to_datetime(df["trans_date_trans_time"], errors='coerce').dt.hour
        df.drop(columns=["trans_date_trans_time"], inplace=True, errors='ignore')

        # HANDLE CATEGORY COLUMN
        df["category"] = df["category"].astype(str).str.replace(",", " -")
        category_dummies = pd.get_dummies(df["category"], prefix="TXNctg")
        df = pd.concat([df, category_dummies], axis=1)
        df.drop(columns=["category"], inplace=True, errors='ignore')

        # HANDLE STATE COLUMN
        df["state"] = df["state"].astype(str)
        state_dummies = pd.get_dummies(df["state"], prefix="state")
        df = pd.concat([df, state_dummies], axis=1)
        df.drop(columns=["state"], inplace=True, errors='ignore')

        # HANDLE JOB CATEGORIES
        jobcat_path = "c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/job_categories.csv"
        jobcat_df = pd.read_csv(jobcat_path)

        df["job"] = df["job"].astype(str).str.replace(",", " -")
        job_category_pairs = []
        for category in jobcat_df.columns:
            jobs_in_cat = jobcat_df[category].dropna().unique()
            job_category_pairs.extend([(job, category) for job in jobs_in_cat])

        job_category_map_df = pd.DataFrame(job_category_pairs, columns=["job", "job_category"])
        df = df.merge(job_category_map_df, on="job", how="left")

        jobcat_dummies = pd.get_dummies(df["job_category"], prefix="JOBctg")
        df = pd.concat([df, jobcat_dummies], axis=1)
        df.drop(columns=["job_category", "job"], inplace=True, errors='ignore')

        # DOB COLUMN: CREATE DECADE FLAGS
        df["dob_year"] = pd.to_datetime(df["dob"], errors='coerce').dt.year
        decades = list(range(1920, 2010, 10))
        for start_year in decades:
            col_name = f"dob_{str(start_year)[2:]}s"
            df[col_name] = np.where(
                (df["dob_year"] >= start_year) & (df["dob_year"] < start_year + 10), 1, 0
            )
        df.drop(columns=["dob_year", "dob"], inplace=True, errors='ignore')

        # CALCULATE DISTANCE BETWEEN CUSTOMER & MERCHANT (Haversine Formula)
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c

        df["distance"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

        # USER BEHAVIORAL METRICS
        user_metrics = df.groupby("cc_num").agg(
            avg_txn_amt=("amt", "mean"),
            stddev_txn_amt=("amt", "std"),
            avg_txn_time=("txn_time", "mean"),
            avg_merchant_distance=("distance", "mean")
        ).round(2).reset_index()

        df = df.merge(user_metrics, on="cc_num", how="left")

        # DROP TEMP COLUMNS
        df.drop(columns=["cc_num", "lat", "long", "merch_lat", "merch_long"], inplace=True, errors='ignore')

        # REORDER COLUMNS
        if "is_fraud" in df.columns:
            other_cols = [c for c in df.columns if c != "is_fraud"]
            df = df[["is_fraud"] + other_cols]

        # OUTPUT PATH
        output_file_path = os.path.join("Uploaded_Datasets", "Processed", f"Processed_{os.path.basename(input_file_path)}")

        df.to_csv(output_file_path, index=False)
        print(f"✅ Successfully processed and saved: {output_file_path}")
        return output_file_path

    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None
