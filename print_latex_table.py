import os 
import argparse
import re 
import numpy as np 
import pandas as pd 
import collections

data = pd.read_csv("raw_results.csv",index_col=0).replace(np.nan,"")   

str_table = '''aemb  & 20603 & 29 & 592 & 
divider  & 10334 & 71 & 153 & 
ethernet  & 82803 & 66 & 533 & 
fdct  & 546878 & 27 & 617 & 
fpu  & 82938 & 27 & 367 & 
mips789  & 55248 & 68 & 1143 & 
rsdecoder  & 14890 & 72 & 1415 & 
scam_core & 1315446 & 66 & 444  & 
spi  & 2536 & 60 & 242 & 
vga  & 44579 & 38 & 933 & 
mean  &   &   &   & '''

table = [[0]*16 for i in range(11)]

rows = str_table.split("\n")
for i in range(len(rows)):
	rows[i] = rows[i].split("&")
	for j in range(len(rows[i])):
		table[i][j] = rows[i][j]
designs = [row[0].strip() for row in rows]

for i in data.index:
	design = data.at[i,"design"]
	idx = designs.index(design)
	predictor = data.at[i,"predictor"]
	prec = data.at[i,"mean_precision"]
	rec = data.at[i,"mean_recall"]
	f1 = data.at[i,"mean_fscore"]
	size_err = data.at[i,"median_size_err"]
	folds = data.at[i,"folds"]

	if data.at[i,"sample_type"] == "solver" and data.at[i,"sample_size"] == 0.5:
		if predictor == "suspect2vec" and not (data.at[i,"lambd"] == 0 and data.at[i,"dim"] == 20):
			continue

		if folds== 1e12:
			offset = int(predictor == "DATE")
			table[idx][4+offset] = prec 
			table[idx][6+offset] = rec 
			table[idx][8+offset] = f1 
			table[idx][10+offset] = size_err

		elif folds in [2,5]:
			offset = 2*int(folds == 5) + int(predictor == "DATE")
			table[idx][12+offset] = f1 

# means 
for i in range(4,16):
	table[-1][i] = np.mean([row[i] for row in table[:-1]])

for i in range(len(table)):
	strs = table[i][:4]
	for j in range(4,16):
		strs.append("%.3f" %(table[i][j]))
	if i == len(rows)-1:
		print "\\hline"
	row = " & ".join(strs).replace("_","\_") + "\\\\"
	print(row)

