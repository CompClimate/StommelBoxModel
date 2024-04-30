import csv


def columns_to_csv_(csv_filepath, colnames, cols):
    """Writes a sequence of sequences of equal length to a csv.

    Args:
        `colnames`: The column names.
        `cols`: The values of the column entries.
    """
    with open(csv_filepath, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(colnames)
        writer.writerows(list(zip(*cols)))
