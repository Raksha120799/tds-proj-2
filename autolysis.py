import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

def main():
    # Check if AIPROXY_TOKEN is set
    if "AIPROXY_TOKEN" not in os.environ:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)
    
    # Get token from environment variable
    openai.api_key = os.environ["AIPROXY_TOKEN"]

    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_filename>")
        sys.exit(1)

    csv_filename = sys.argv[1]

    # Check if the file exists
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        sys.exit(1)

    # Load the dataset
    try:
        print(f"Loading dataset: {csv_filename}")
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    print("Dataset loaded successfully.")
    print("Performing analysis...")

    # Example Analysis: Summary Statistics
    summary = df.describe(include='all').to_string()
    missing_values = df.isnull().sum().to_string()

    # Visualization: Missing Values
    try:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.savefig("missing_values.png")
        plt.close()
        print("Missing values heatmap created successfully.")
    except Exception as e:
        print(f"Error creating missing values heatmap: {e}")

    # Clean the dataset: Handle missing values and non-numeric columns
    df_cleaned = df.copy()

    # Fill missing values with the column mean (for numeric columns only)
    df_cleaned = df_cleaned.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

    # Remove non-numeric columns before correlation analysis
    numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

    # Correlation Matrix (only for numeric columns)
    if numeric_df.shape[1] > 1:
        try:
            correlation_matrix = numeric_df.corr()

            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.savefig("correlation.png")
            plt.close()
            print("Correlation matrix heatmap created successfully.")
        except Exception as e:
            print(f"Error creating correlation matrix heatmap: {e}")
    else:
        print("Not enough numeric data for correlation analysis.")

    # Generate README.md using LLM
    try:
        print("Generating README.md using GPT-4o-Mini...")
        llm_prompt = (
            f"Dataset Summary:\n{summary}\n\n"
            f"Missing Values:\n{missing_values}\n\n"
            f"Write a story narrating the analysis results and insights."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": llm_prompt}]
        )
        story = response.choices[0].message["content"]
        with open("README.md", "w") as f:
            f.write(story)
        print("README.md generated successfully.")
    except Exception as e:
        print(f"Error generating README.md: {e}")

    print("Analysis complete.")

    # Ensure files are saved
    if os.path.isfile("README.md"):
        print("README.md created successfully.")
    else:
        print("Error: README.md not created.")

    if os.path.isfile("missing_values.png"):
        print("Missing values heatmap created successfully.")
    else:
        print("Error: missing_values.png not created.")

    if os.path.isfile("correlation.png"):
        print("Correlation matrix heatmap created successfully.")
    else:
        print("Error: correlation.png not created.")

if __name__ == "__main__":
    main()