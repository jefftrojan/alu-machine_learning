#!/bin/bash

# check if there are any files in the dir
if [ -z "$(ls -A .)" ]; then
  echo "No files to commit."
  exit 1
fi

# loop through each file in the dir
for file in *; do
  if [ -f "$file" ]; then
    # Add the file to the staging area
    git add "$file"
    
    git commit -m " $file"
    
    echo "Committed $file"
  fi
done
