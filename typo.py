import sys
import random

fin_name = sys.argv[1]
probs = [.001, .005, .01]

for i in range(3):
    print probs[i]
    fin = open(fin_name)
    fout_name = fin_name[:-1]  + str(i + 4)
    fout = open(fout_name, "w")
    print fout_name
    count = 0
    for line in fin:
        for w in line.split():
            if random.random() < probs[i]:
                fout.write(w[1:])
                count = count + 1
            else:
                fout.write(w)
            fout.write(" ")
    print count
    fout.close
    fin.close



