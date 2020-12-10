import sys
import pandas as pd
import csv


def readFile(filePath):
    # Define file extension from path

    # Read in file, determine whether a pkl or txt/csv
    try:

        # Detect delimiter
        sniffer = csv.Sniffer()
        sniffer.preferred = [',', '|', ';', ':', '~']
        csvFile = open(filePath, 'rt')
        row1 = []
        for row in csv.reader(csvFile, delimiter="\t"):
            row1 = row
            break
        csvFile.close()
        dialect = sniffer.sniff(str(row1))
        sepType = dialect.delimiter

        if sepType not in {",", "|", ";", ":", "~"}:
            print("Invalid delimiter")
            sys.stdout.flush()
            return

        # Read in pandas data frame from csv file
        df = pd.read_csv(filePath, sep=sepType)
    except pd.errors.ParserError:
        print("Invalid file")
        sys.stdout.flush()
        return
    except IOError:
        print("File not found")
        sys.stdout.flush()
        return

    return df
