cat *_* | tr -cd '[:print:]' | tr -s '[:space:][:punct:]' '\n' | sort -u > dict.txt