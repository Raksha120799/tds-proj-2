# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "httpx",
#   "chardet",
#   "python-dotenv",
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import httpx
import chardet

# Force non-interactive matplotlib backend
matplotlib.use('Agg')

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Replace with actual token for Colab if necessary.

if not AIPROXY_TOKEN:
    raise ValueError("API token not set. Please set AIPROXY_TOKEN in the environment.")

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    analysis = {
        'summary': df.describe(include='all').to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
    }
    return analysis

def visualize_data(df, output_dir):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        output_path = os.path.join(output_dir, f'{column}_distribution.png')
        plt.savefig(output_path)
        print(f"Saved plot: {output_path}")
        plt.close()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main():
    # For Colab, use an interactive method to upload files
    from google.colab import files

    # Prompt user to upload file
    print("Please upload your dataset CSV file:")
    uploaded = files.upload()  # Upload file via Colab interface

    # Get the file path
    file_path = next(iter(uploaded.keys()))  # First uploaded file
    print(f"Loaded file: {file_path}")

    # Define output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(file_path)

    # Analyze data
    analysis = analyze_data(df)

    # Visualize data
    visualize_data(df, output_dir)

    # Generate narrative
    narrative = generate_narrative(analysis)

    # Save narrative
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(narrative)
    print(f"Narrative saved to: {readme_path}")

    # Download the outputs
    print("Preparing outputs for download...")
    files.download(readme_path)
    for column in df.select_dtypes(include=['number']).columns:
        plot_path = os.path.join(output_dir, f'{column}_distribution.png')
        files.download(plot_path)

if __name__ == "__main__":
    main()