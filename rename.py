import csv
import os
import shutil


labels = {}

# with open('trainLabels.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         labels[f'{row["image"]}.tiff'] = row["level"]

# for f in os.walk("data/train128px"):
#     for file in f[2]:
#         print(f"name: {file} label: {labels[file]}")
#         shutil.move(f"data/train128px/{file}", f"data/labeled_train128px/{labels[file]}/{file}")


with open('testLabels.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        labels[f'{row["image"]}.tiff'] = row["level"]

for f in os.walk("data/test128px"):
    for file in f[2]:
        print(f"name: {file} label: {labels[file]}")
        shutil.move(f"data/test128px/{file}", f"data/labeled_test128px/{labels[file]}/{file}")
