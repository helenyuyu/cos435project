#!/bin/bash

# mkdir termfreqs

for f in dataset/*_*
do
	f2=$f'a'
	echo $f2
	tr -s '[:space:][:punct:]' '\n' < $f | sort | uniq -c > $f2
done

# mv dataset/*_*a dataset/tf