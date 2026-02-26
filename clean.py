import pandas as pd

# Path to the uploaded file
file_path = './players_match_data_5000_100.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert 'gameStartTimestamp' and 'gameEndTimestamp' to datetime format
df['gameStartTimestamp'] = pd.to_datetime(df['gameStartTimestamp'], unit='ms')
df['gameEndTimestamp'] = pd.to_datetime(df['gameEndTimestamp'], unit='ms')
df['gameCreation'] = pd.to_datetime(df['gameCreation'], unit='ms')
# Save the updated DataFrame to a new CSV file
output_file_path = './updated_participants_with_match_data.csv'
df.to_csv(output_file_path, index=False)
