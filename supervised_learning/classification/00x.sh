#!/bin/bash

# Check if there are any files in the directory
if [ -z "$(ls -A .)" ]; then
  echo "No files to commit."
  exit 1
fi

# Loop through each file in the directory
for file in *; do
  # Check if it is a file (and not a directory)
  if [ -f "$file" ]; then
    # Add the file to the staging area
    git add "$file"
    
    # Commit the file with a specific commit message
    git commit -m "Added file $file"
    
    # Print a message indicating the file has been committed
    echo "Committed $file"
  fi
done
