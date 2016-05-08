import os
import glob
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

k = int(sys.argv[1])
dir = "dataset"
files = os.listdir(dir)
fileToShingles = dict()
# get all shingles
for file in files:
    with open(dir + "/" + file, 'r') as content_file:
        text = content_file.read()
        words = text.split()
        shingles = set()
        for i in range(len(words)-k):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        fileToShingles[file] = shingles

probs = []
labels = []
origs = [os.path.basename(x) for x in glob.glob('dataset/*_0')]
for file in files:
    if not file in origs:
        for origFile in origs:
            shingles1 = fileToShingles[file]
            shingles2 = fileToShingles[origFile];
            resemblance = (len(shingles1.intersection(shingles2)) + 0.0)/len(shingles1.union(shingles2))
            probs.append(resemblance)
            if file[:-1] == origFile[:-1]:
                labels.append(1)
            else:
                labels.append(0)

for file in files:
    origFile = file[:-1] + "0"
    shingles1 = fileToShingles[file]
    shingles2 = fileToShingles[origFile];
    resemblance = (len(shingles1.intersection(shingles2)) + 0.0)/len(shingles1.union(shingles2))
    print "resemblance between " + file + " and " + origFile + " = " + str(resemblance)

precision, recall, _ = precision_recall_curve(labels, probs)
average_precision = average_precision_score(labels, probs)
plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall for Shingling k = {}'.format(str(k)))
plt.show()
print average_precision
