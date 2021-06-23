#! /usr/bin/env python
import csv

f = open(input(), 'r')  # ファイル名を標準入力
tsv = csv.reader(f, delimiter = '\t')
for row in tsv:
    print(row)
f.close()
