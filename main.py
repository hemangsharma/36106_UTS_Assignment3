import pandas as pd

def process_transactions_files():
    for i in range(132):
        # Generate the filename
        filename = f'transactions_{i}.csv'
        
        # Load CSV file
        df = pd.read_csv(filename, delimiter='|')

        # Print the DataFrame
        print(f'{filename}:')
        print(df)

        # Save the DataFrame to a new CSV file
        new_filename = f'new_transactions_{i}.csv'
        df.to_csv(new_filename, index=False)

process_transactions_files()
