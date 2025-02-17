#!/usr/bin/env python3
"""
Docker-ready script for processing employee and salary data.
Usage:
    python script.py --employee-data data/Employee_dataset.csv --salaries-data data/Employee_salaries.csv
"""

import pandas as pd
import os
import argparse

def load_and_clean_data(employee_file, salaries_file):
    # LOADING DATASETS
    employee_dataset = pd.read_csv(employee_file)
    salaries_dataset = pd.read_csv(salaries_file)

    # REMOVE DUPLICATES based on "jobId"
    employee_dataset = employee_dataset.drop_duplicates(subset=["jobId"])
    # DROP NULL COLUMNS (Columns that are entirely null)
    employee_dataset = employee_dataset.dropna(axis=1, how='all')
    # DROP ROWS where "jobId" is null
    employee_dataset = employee_dataset.dropna(subset=["jobId"])

    # REMOVE DUPLICATES based on "jobId"
    salaries_dataset = salaries_dataset.drop_duplicates(subset=["jobId"])
    # DROP NULL COLUMNS (Columns that are entirely null)
    salaries_dataset = salaries_dataset.dropna(axis=1, how='all')
    # DROP ROWS where "jobId" is null
    salaries_dataset = salaries_dataset.dropna(subset=["jobId"])

    # Impute "NONE" for missing values in categorical columns
    categorical_cols = ["companyId", "jobRole", "education", "major", "industry"]
    employee_dataset[categorical_cols] = employee_dataset[categorical_cols].fillna("NONE")

    # Impute median for numerical columns
    numerical_cols = ["yearsExperience", "distanceFromCBD"]
    for col in numerical_cols:
        median_val = employee_dataset[col].median()
        employee_dataset[col] = employee_dataset[col].fillna(median_val)

    # Merge Datasets
    merged_df = pd.merge(employee_dataset, salaries_dataset, on="jobId", how="left")

    # Impute median for the salary column
    median_salary = merged_df["salaryInThousands"].median()
    merged_df["salaryInThousands"] = merged_df["salaryInThousands"].fillna(median_salary)

    # Replace "0" values in salaryInThousands with the median value
    merged_df.loc[merged_df["salaryInThousands"] == 0, "salaryInThousands"] = median_salary

    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Process employee and salary datasets.")
    parser.add_argument(
        "--employee-data",
        type=str,
        default="data/Employee_dataset.csv",
        help="Path to the employee dataset CSV file"
    )
    parser.add_argument(
        "--salaries-data",
        type=str,
        default="data/Employee_salaries.csv",
        help="Path to the employee salaries CSV file"
    )
    args = parser.parse_args()

    # Check if data files exist; if running inside Docker, ensure the host directory is mounted.
    if not os.path.exists(args.employee_data):
        print(f"Employee data file not found: {args.employee_data}")
        return

    if not os.path.exists(args.salaries_data):
        print(f"Salaries data file not found: {args.salaries_data}")
        return

    merged_df = load_and_clean_data(args.employee_data, args.salaries_data)
    print(merged_df.head())
    # Optionally, save the merged data to a file
    # merged_df.to_csv("data/merged_output.csv", index=False)

if __name__ == "__main__":
    main()
