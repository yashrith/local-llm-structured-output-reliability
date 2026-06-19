import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure cohesive, high-contrast engineering style guidelines
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 15,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa'
})

csv_file = "syntax_rot_multi_temp_results.csv"

if not os.path.exists(csv_file):
    print(f"❌ Error: Cannot locate '{csv_file}'. Ensure it is in the active directory.")
    exit()

# Load evaluation logs
df = pd.read_csv(csv_file)
df['is_failure'] = df['validation_status'].apply(lambda x: 1 if x in ['FAIL', 'SYSTEM_ERROR'] else 0)

# Define clean hardware labels for mapping 
model_mapping = {
    'llama3.2': 'Llama 3.2 (3B Baseline)',
    'llama3.1:8b': 'Llama 3.1 (8B Standard)',
    'qwen2.5:14b': 'Qwen 2.5 (14B Boundary)'
}
df['Model_Tier'] = df['model'].map(model_mapping)
unique_models = [model_mapping[m] for m in ['llama3.2', 'llama3.1:8b', 'qwen2.5:14b']]

# Custom 95th Percentile Aggregator
def p95_latency(x):
    return np.percentile(x, 95)

# Calculate explicit production metrics
metrics = df.groupby(['Model_Tier', 'temperature']).agg(
    Total_Requests=('loop_index', 'count'),
    Failure_Rate_Pct=('is_failure', lambda x: x.mean() * 100),
    Mean_Latency_Sec=('latency_seconds', 'mean'),
    P95_Latency_Sec=('latency_seconds', p95_latency),
    std_Dev_Latency=('latency_seconds', 'std')
).reset_index()

print("📊 Compiled Multi-Dimensional MLOps Engineering Metrics Grid.")

# =====================================================================
# FIGURE 1: RELIABILITY MATRIX HEATMAP
# =====================================================================
plt.figure(figsize=(8, 5))
pivot_heatmap = metrics.pivot(index='Model_Tier', columns='temperature', values='Failure_Rate_Pct')
pivot_heatmap = pivot_heatmap.reindex(unique_models)

ax = sns.heatmap(
    pivot_heatmap, 
    annot=True, 
    fmt=".1f", 
    cmap="YlOrRd", 
    cbar_kws={'label': 'Validation Failure Rate (%)'},
    linewidths=1.5,
    linecolor='white',
    annot_kws={"weight": "bold", "size": 12}
)
plt.title("Structural JSON Schema Validation Failure Matrix", weight='bold', pad=15)
plt.xlabel("Sampling Temperature ($\tau$)", labelpad=10)
plt.ylabel("Model Tier & Scale Parameters", labelpad=10)
plt.tight_layout()
plt.savefig("reliability_matrix_heatmap.png", dpi=300)
plt.close()
print("✅ Rendered: 'reliability_matrix_heatmap.png'")

# =====================================================================
# FIGURE 2: FACETED LATENCY DISTRIBUTIONS (CORRECTED: sharex=True)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
fig.suptitle("Isolated Multi-Panel Generation Latency Windows", weight='bold', y=0.98)

# Loop explicitly across tiers to assign distinct, legible y-axis constraints
for idx, model_name in enumerate(unique_models):
    sub_df = df[df['Model_Tier'] == model_name]
    
    # Draw boxplot inside target subplot panel
    sns.boxplot(
        data=sub_df,
        x='temperature',
        y='latency_seconds',
        ax=axes[idx],
        color=sns.color_palette("muted")[idx],
        width=0.5,
        flierprops={"marker": "o", "markerfacecolor": "gray", "alpha": 0.5, "markersize": 4}
    )
    
    # Overlay a trendline connecting averages across temperature profiles
    sns.pointplot(
        data=sub_df,
        x='temperature',
        y='latency_seconds',
        ax=axes[idx],
        color='black',
        errorbar=None,
        markers='d',
        linestyles='--'
    )
    
    axes[idx].set_title(model_name, weight='semibold', pad=10)
    axes[idx].set_xlabel("Temperature ($\tau$)", labelpad=8)
    axes[idx].set_ylabel("Latency (Seconds)" if idx == 0 else "", labelpad=5)
    
    # Dynamically optimize limits with padding to show variance explicitly without compression
    y_min = sub_df['latency_seconds'].min() * 0.9
    y_max = sub_df['latency_seconds'].max() * 1.1
    axes[idx].set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig("faceted_latency_distributions.png", dpi=300)
plt.close()
print("✅ Rendered: 'faceted_latency_distributions.png'")

# =====================================================================
# FIGURE 3: PRODUCTION SLA JITTER & TAIL LATENCY MAP
# =====================================================================
plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for idx, model_name in enumerate(unique_models):
    model_data = metrics[metrics['Model_Tier'] == model_name]
    
    # 1. Plot Mean Ingestion Performance
    plt.plot(
        model_data['temperature'], 
        model_data['Mean_Latency_Sec'], 
        label=f"{model_name} [Mean]", 
        color=colors[idx], 
        linestyle='-', 
        marker=markers[idx], 
        linewidth=2
    )
    
    # 2. Plot P95 Production SLA Tail Latency Performance
    plt.plot(
        model_data['temperature'], 
        model_data['P95_Latency_Sec'], 
        label=f"{model_name} [P95 Tail]", 
        color=colors[idx], 
        linestyle=':', 
        marker=markers[idx], 
        alpha=0.7,
        linewidth=1.5
    )
    
    # Shade the resource variance buffer (Mean to P95)
    plt.fill_between(
        model_data['temperature'],
        model_data['Mean_Latency_Sec'],
        model_data['P95_Latency_Sec'],
        color=colors[idx],
        alpha=0.08
    )

plt.title("Production Ingestion Stability: Mean vs. P95 Tail Latency", weight='bold', pad=15)
plt.xlabel("Sampling Temperature ($\tau$)", labelpad=10)
plt.ylabel("System Ingestion Latency (Seconds)", labelpad=10)
plt.xticks(metrics['temperature'].unique())
plt.legend(title="Performance Metrics", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("production_sla_tail_latency.png", dpi=300)
plt.close()
print("✅ Rendered: 'production_sla_tail_latency.png'")

print("\n🚀 Production visualization architecture generated cleanly.")