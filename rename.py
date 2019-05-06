import csv
import os
import shutil

dimension = '128px'
labels = {}

# move  resized images to dir structure needed by tf
for dataset in ['test', 'train']:
    with open(f'{dataset}Labels.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            labels[f'{row["image"]}.tiff'] = row["level"]

    for f in os.walk(f"data/train{dimension}"):
        for file in f[2]:
            shutil.move(f"data/{dataset}{dimension}/{file}", f"data/labeled_{dataset}{dimension}/{labels[file]}/{file}")
