import csv

# f = open(input(), 'r')  # ファイル名を標準入力
f = open("3588797130_a.tsv",'r')
tsv = csv.reader(f, delimiter = '\t')
list = []
for row in tsv:
    i0 = row[0]
    if row[1] == "鬼":
        i1 = 0
    else:
        i1 = 1

    print(i1)

f.close()

