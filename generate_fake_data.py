import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- CONFIGURATION ---
NUM_USERS = 20
NUM_DAYS = 60
START_DATE = datetime.now() - timedelta(days=NUM_DAYS)

# --- USER PROFILES ---
users = [f"emp_{i:03d}" for i in range(1, NUM_USERS + 1)]

departments = {
    "emp_001": "Engineering",
    "emp_002": "Engineering",
    "emp_003": "Engineering",
    "emp_004": "Sales",
    "emp_005": "Sales",
    "emp_006": "Sales",
    "emp_007": "Marketing",
    "emp_008": "Marketing",
    "emp_009": "Marketing",
    "emp_010": "HR",
    "emp_011": "HR",
    "emp_012": "Finance",
    "emp_013": "Finance",
    "emp_014": "Engineering",
    "emp_015": "Engineering",
    "emp_016": "Sales",
    "emp_017": "Marketing",
    "emp_018": "IT",
    "emp_019": "IT",
    "emp_020": "Operations"
}

# --- GENERATE DATA ---
rows = []

for user in users:
    # Random work pattern for each user
    work_hours = random.choice([7, 8, 9, 10])  # Average work hours
    login_time = random.choice([8, 9, 10])  # Start time
    
    for day in range(NUM_DAYS):
        date = START_DATE + timedelta(days=day)
        is_weekend = date.weekday() >= 5
        
        # Skip some weekends (50% chance of working on weekend)
        if is_weekend and random.random() > 0.3:
            continue
        
        # --- LOGIN EVENT ---
        login_hour = login_time + random.randint(-1, 1)
        login_minute = random.randint(0, 59)
        login_timestamp = date.replace(hour=login_hour, minute=login_minute)
        
        rows.append({
            "user_id": user,
            "department": departments.get(user, "Other"),
            "timestamp": login_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "login"
        })
        
        # --- LOCK EVENTS (2-5 per day) ---
        num_locks = random.randint(2, 5)
        for i in range(num_locks):
            lock_hour = random.randint(10, 17)
            lock_minute = random.randint(0, 59)
            lock_timestamp = date.replace(hour=lock_hour, minute=lock_minute)
            
            rows.append({
                "user_id": user,
                "department": departments.get(user, "Other"),
                "timestamp": lock_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "lock"
            })
            
            # Unlock 5-30 minutes later
            unlock_delta = random.randint(5, 30)
            unlock_timestamp = lock_timestamp + timedelta(minutes=unlock_delta)
            
            if unlock_timestamp < date.replace(hour=18, minute=0):
                rows.append({
                    "user_id": user,
                    "department": departments.get(user, "Other"),
                    "timestamp": unlock_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "unlock"
                })
        
        # --- LOGOUT EVENT ---
        logout_hour = login_hour + work_hours + random.randint(-1, 2)
        logout_minute = random.randint(0, 59)
        logout_timestamp = date.replace(hour=min(logout_hour, 22), minute=logout_minute)
        
        # Late work (after 8 PM)
        if random.random() < 0.2:  # 20% chance of late work
            logout_hour = random.randint(20, 23)
            logout_timestamp = date.replace(hour=logout_hour, minute=random.randint(0, 59))
        
        rows.append({
            "user_id": user,
            "department": departments.get(user, "Other"),
            "timestamp": logout_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "logout"
        })

# --- CREATE DATAFRAME ---
df = pd.DataFrame(rows)

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# --- SAVE TO CSV ---
filename = "fake_login_data.csv"
df.to_csv(filename, index=False)

print(f"✅ Generated {len(df)} rows")
print(f"📁 Saved to: {filename}")
print(f"👥 Users: {df['user_id'].nunique()}")
print(f"📅 Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"🏢 Departments: {', '.join(df['department'].unique())}")

# --- DATA STATISTICS ---
print("\n--- Event Type Counts ---")
print(df["event_type"].value_counts())

print("\n--- Events per User ---")
print(df.groupby("user_id").size().sort_values(ascending=False).head())

print("\n--- Sample Data ---")
print(df.head())