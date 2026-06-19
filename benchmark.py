import os
import time
import json
import pandas as pd
from typing import List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError
import ollama

# ==========================================
# 1. DEFINE THE EXPECTED JSON STRUCTURE
# ==========================================
class UserProfile(BaseModel):
    user_id: int = Field(description="Unique incremental integer starting from 1")
    username: str = Field(description="A random lowercase alphanumeric username")
    email: str = Field(description="A valid email address ending in .com")
    account_active: bool = Field(description="True or False flag")
    score: float = Field(description="A floating-point decimal performance score between 0.0 and 100.0")

class DatabasePayload(BaseModel):
    profiles: List[UserProfile]

# Helper function to convert raw seconds into clean HHh MMm SSs formatting
def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

# ==========================================
# 2. CONFIGURATION & BENCHMARK SETUP
# ==========================================
MODELS_TO_TEST = ['llama3.2', 'llama3.1:8b', 'qwen2.5:14b']
TEMPERATURES_TO_TEST = [0.0, 0.3, 0.7, 1.0]
NUM_LOOPS = 100

SYSTEM_PROMPT = """You are a backend data ingestion utility. You MUST output data strictly adhering to the JSON schema requested.
Do not provide any conversational preamble, markdown formatting, backticks (```json), or postscript. 
Output raw, minified valid JSON only."""

USER_PROMPT = """Generate a valid JSON object matching this structural definition exactly:
{
  "profiles": [
    {"user_id": 1, "username": "alpha99", "email": "test1@domain.com", "account_active": true, "score": 88.5},
    {"user_id": 2, "username": "beta22", "email": "test2@domain.com", "account_active": false, "score": 42.1}
  ]
}
Provide exactly 3 profiles in your array list."""

results_log = []

# Capture total experiment tracking markers
script_start_time_raw = time.time()
script_start_clock = datetime.now().strftime("%H:%M:%S")

print(f"🚀 Starting Multi-Temperature Syntax Rot Benchmark Engine...")
print(f"⏰ Global Execution Started at Wall-Clock Time: {script_start_clock}")
print(f"📋 Matrix Size: {len(MODELS_TO_TEST)} models x {len(TEMPERATURES_TO_TEST)} temperatures x {NUM_LOOPS} loops = {len(MODELS_TO_TEST) * len(TEMPERATURES_TO_TEST) * NUM_LOOPS} total test points.")

model_durations = {}

# ==========================================
# 3. RUN THE MULTI-DIMENSIONAL STRESS TEST
# ==========================================
for model in MODELS_TO_TEST:
    print(f"\n==========================================")
    print(f"🏗️  STARTING EVALUATION FOR MODEL: {model}")
    print(f"==========================================")
    
    # Track the exact duration of each individual model execution
    model_start_time_raw = time.time()
    
    for temp in TEMPERATURES_TO_TEST:
        print(f"\n--- Testing Temp Setting: {temp} ---")
        
        for loop in range(1, NUM_LOOPS + 1):
            # Capture precise micro wall-clock timestamps per loop
            loop_start_clock = datetime.now().strftime("%H:%M:%S")
            start_time = time.time()
            
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': USER_PROMPT}
                    ],
                    options={
                        'temperature': temp
                    }
                )
                
                raw_output = response['message']['content'].strip()
                generation_time = time.time() - start_time
                loop_end_clock = datetime.now().strftime("%H:%M:%S")
                
                if raw_output.startswith("```"):
                    raw_output = raw_output.strip("```").replace("json", "", 1).strip()
                
                try:
                    parsed_json = json.loads(raw_output)
                    DatabasePayload(**parsed_json)
                    status = "PASS"
                    error_message = ""
                    print(f"[{loop_start_clock} -> {loop_end_clock}] Model: {model} | Temp: {temp} | Loop: [{loop}/{NUM_LOOPS}] -> ✅ Passed ({generation_time:.2f}s)")
                except (json.JSONDecodeError, ValidationError) as ve:
                    status = "FAIL"
                    error_message = str(ve).replace('\n', ' ')
                    print(f"[{loop_start_clock} -> {loop_end_clock}] Model: {model} | Temp: {temp} | Loop: [{loop}/{NUM_LOOPS}] -> ❌ FAILED Syntax check ({generation_time:.2f}s)")
                    
            except Exception as system_err:
                generation_time = time.time() - start_time
                loop_end_clock = datetime.now().strftime("%H:%M:%S")
                status = "SYSTEM_ERROR"
                raw_output = ""
                error_message = str(system_err)
                print(f"[{loop_start_clock} -> {loop_end_clock}] Model: {model} | Temp: {temp} | Loop: [{loop}/{NUM_LOOPS}] -> ⚠️ Local Engine Error ({generation_time:.2f}s)")
                
            # Append wall-clock metrics to row indices for the dataset
            results_log.append({
                "model": model,
                "temperature": temp,
                "loop_index": loop,
                "start_time_clock": loop_start_clock,
                "end_time_clock": loop_end_clock,
                "latency_seconds": generation_time,
                "validation_status": status,
                "error_type": error_message,
                "raw_response": raw_output
            })
            
    # Calculate and store individual model time tracking profiles
    model_end_time_raw = time.time()
    model_elapsed_seconds = model_end_time_raw - model_start_time_raw
    model_durations[model] = model_elapsed_seconds
    print(f"\n⏱️ Completed Tier {model}. Total Duration: {format_duration(model_elapsed_seconds)}")

# Record complete macro execution timelines
script_end_time_raw = time.time()
script_end_clock = datetime.now().strftime("%H:%M:%S")
total_script_elapsed_seconds = script_end_time_raw - script_start_time_raw

# ==========================================
# 4. EXPORT DATASET TO CSV
# ==========================================
df = pd.DataFrame(results_log)
output_filename = "syntax_rot_multi_temp_results.csv"
df.to_csv(output_filename, index=False)

print(f"\n\n==========================================")
print(f"✅ BENCHMARK RUN COMPLETES SUCCESSFULLY")
print(f"==========================================")
print(f"📊 Comprehensive evaluation dataset saved to: {os.path.abspath(output_filename)}")
print(f"⏰ Execution Timeline: Started at {script_start_clock} | Ended at {script_end_clock}")
print(f"⏳ Total Complete Execution Duration: {format_duration(total_script_elapsed_seconds)}")

print("\n⏱️ SUMMARY OF INDIVIDUAL MODEL TOTAL DURATIONS:")
for target_model, total_seconds in model_durations.items():
    print(f"  ▪️ {target_model}: {format_duration(total_seconds)}")

# Initial Pivot Summary Report Table
print("\n📋 INITIAL MATRIX SUMMARY (Failure vs. Success counts):")
summary = df.groupby(['model', 'temperature', 'validation_status']).size().unstack(fill_value=0)
print(summary)