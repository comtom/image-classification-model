import csv
import os
import shutil


labels = {}

with open('trainLabels.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        labels[f'{row["image"]}.tiff'] = row["level"]



for f in os.walk("data/test"):
    for file in f[2]:
        print(f"name: {file} label: {labels[file]}")
        shutil.move(f"data/test/{file}", f"data/labeled_data/{labels[file]}/{file}")
