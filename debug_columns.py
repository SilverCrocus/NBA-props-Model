import pandas as pd

betting_df = pd.read_csv('data/results/backtest_2024_25_FIXED_V2_betting.csv')
print("Original columns:")
print(betting_df.columns.tolist())
print(f"\nDuplicates: {betting_df.columns.duplicated().sum()}")

# Remove duplicates
betting_df = betting_df.loc[:, ~betting_df.columns.duplicated()]
print("\nAfter removing duplicates:")
print(betting_df.columns.tolist())

# Uppercase
betting_df.columns = betting_df.columns.str.upper()
print("\nAfter uppercase:")
print(betting_df.columns.tolist())
print(f"\nDuplicates after uppercase: {betting_df.columns.duplicated().sum()}")

# Check for PLAYER_NAME
print("\nPLAYER_NAME column info:")
print(betting_df['PLAYER_NAME'].head())
