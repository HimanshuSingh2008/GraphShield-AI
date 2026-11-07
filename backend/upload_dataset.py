import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

dataset_record = {
    "dataset_name": "PaySim Fraud Detection",
    "storage_path": None,
    "total_transactions": 200000,
    "total_accounts": None
}

response = supabase.table("datasets").insert(
    dataset_record
).execute()

print("Dataset metadata saved to Supabase")
print(response)

