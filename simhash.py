import sys
import os
import json
import csv
import random
from sets import Set
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


# maps each term in file fin to an f-dimensional vector over {-1, 1}
def termvectors(terms, f):
    t = {}
    for term in terms:
        h = bin(random.getrandbits(f))[2:].zfill(f)   # random f-dim bit vector
        t[term.strip()] = [-1 + 2 * int(x)
                           for x in h] 	  # x=0 maps to -1; x=1 maps to 1
    return t


# given the f-dim vector representations of the terms in termvectors
# and the term weights of the documents in docterms,
# returns the f-dim signature vector for the document
def signature(docterms, termvectors, f):
    v = [0 for x in range(f)]
    for term, weight in docterms.items():
        if term in termvectors:
            t = termvectors[term]
            for i in range(f):
                v[i] += weight * t[i]
    return [0 if x < 0 else 1 for x in v]


# all files in dir are formatted with one <freq, term> pair per line
# reads in all files in dir and returns a map from filename to it's
# bag-of-words representation (map of terms to freq/(total terms))
def load_docs(dir):
    allfiles = os.listdir(dir)
    alldocs = {}
    for fn in allfiles:
        term_weights = {}
        with open(dir + "/" + fn, 'r') as fin:
            total = 0.
            for line in fin:
                freq, term = line.split()
                total += int(freq)
                term_weights[term] = int(freq)
        term_weights = {k: v / total for k, v in term_weights.iteritems()}
        alldocs[fn] = term_weights
    return alldocs


# returns the Hamming distance between two binary vectors
# a1 and a2 of equal lengths
# nomalized by dividing by the len of the vectors
def hamdist(a1, a2):
    diffs = 0
    for x1, x2 in zip(a1, a2):
        diffs += x1 ^ x2
    return diffs / float(len(a1))


# given a set of doc names and their signatures, returns a map of
# pairs of documents and the normalized hamming distance between them
# uniq is a boolean flag. When uniq is True, comparedocs only returns unique
# combinations of pairs of documents, with the first document name strictly
# smaller than the second.
# Otherwise, it returns all possible pairs of documents
def comparedocs(sigs, uniq):
    if uniq:
        doc_comps = {
            (a, b): hamdist(sigs[a], sigs[b])
            for b in sigs
            for a in sigs
            if a < b
        }
    else:
        doc_comps = {
            a: {b: hamdist(sigs[a], sigs[b]) for b in sigs}
            for a in sigs
        }
    return doc_comps


# outputs distances between all pairs of documents as a .csv file to fout
def comps_to_csv(sigs, fout):
    comps = comparedocs(sigs, False)
    with open(fout, 'wb') as csvfile:
        dw = csv.DictWriter(csvfile, sorted(comps.keys()))
        dw.writeheader()
        for k, vd in sorted(comps.items()):
            dw.writerow(comps[k])


# comps is a map from a pair of documents to hamming distance between them.
# t is the threshold for the hamming distance
# returns a list of all pairs of documents considered to be near-duplicates.
def dupplicatepairs(comps, t):
    duplicates = []
    for pair, hamdist in comps.items():
        if hamdist < t:
            duplicates += [pair]
    return duplicates


# comps is a map from a pair of documents to hamming distance between them.
# t is the threshold for the hamming distance
# returns whether two documents a and b are near-duplicates
def isdup(a, b, comps, t):
    hamdist = comps[(min(a, b), max(a, b))]
    return hamdist < t


# For our testing purposes:
# returns the set of pairs we labelled as duplicates
# and the full set of pairs we're testing
def testpairs(fns):
    exp_pairs = Set([])
    test_pairs = Set([])
    for fn1 in fns:
        a = fn1 + "_0a"
        for fn2 in fns:
            b = fn2 + "_0a"
            for i in range(1, 8):
                b = b[:-2] + str(i) + "a"
                if fn1 == fn2:
                    exp_pairs.add((min(a, b), max(a, b)))
                test_pairs.add((min(a, b), max(a, b)))
    return exp_pairs, test_pairs


termsfn = "dict.txt"
term_dir = "dataset/tf"

docdists = {}
fs = [128, 96, 64, 32]
for f in fs:
    with open(termsfn, 'r') as termsfile:
        terms = termsfile.readlines()
        t = termvectors(terms, f)
        termsfile.close()

    all_docs = load_docs(term_dir)
    sigs = {}
    for fn, term_weights in all_docs.iteritems():
        sigs[fn] = signature(term_weights, t, f)

    docdists[f] = comparedocs(sigs, True)
    comps_to_csv(sigs, "filecomps_" + str(f) + ".csv")

filestems = ["101666", "102616", "102764", "14832", "38851", "51060",
             "52434", "57110", "59848", "74724"]

exp_dups, test_pairs = testpairs(filestems)

pairs = sorted(test_pairs)

precision = dict()
recall = dict()
average_precision = dict()

for f in fs:
    y_true = np.array([1 if (x in exp_dups) else 0 for x in pairs])
    y_scores = np.array([1 - docdists[f][x] for x in pairs])
    precision[f], recall[f], thresholds = precision_recall_curve(y_true, y_scores)
    average_precision[f] = average_precision_score(y_true, y_scores)
    print precision[f]
    print recall[f]
    print thresholds

    print average_precision

plt.clf()
for f in fs:
    plt.plot(recall[f], precision[f],
             label='Precision-recall curve for f = {0} (area = {1:0.2f})'
                   ''.format(f, average_precision[f]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for SimHash algorithm')
plt.legend(loc="lower right")
plt.show()
