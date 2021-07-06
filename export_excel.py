import csv
import pandas as pd

if __name__ == "__main__":
    """ with open("./test.csv", "w", newline="") as csvfile:
        fieldnames = ['first_col', 'second_col']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'first_col': 'a','second_col': 'b'})
        writer.writerow({'first_col': 'a','second_col': 'b'})
        writer.writerow({'first_col': 'a','second_col': 'b'}) """

    read_file = pd.read_csv (r'./3001_results.csv')
    read_file.to_excel(r'./3001_results.xlsx', index=None, header=True)
