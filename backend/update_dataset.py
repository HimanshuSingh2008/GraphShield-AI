import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")

supabase = create_client(supabase_url, supabase_key)

# Load processed transactions
df = pd.read_csv("ml/data/processed_transactions.csv")

# Calculate unique accounts
total_accounts = pd.concat([
    df["nameOrig"],
    df["nameDest"]
]).nunique()

# Update dataset metadata
response = (
    supabase.table("datasets")
    .update({
        "total_transactions": len(df),
        "total_accounts": int(total_accounts)
    })
    .eq("id", 1)
    .execute()
)

print("Dataset metadata updated successfully!")
print(response.data)