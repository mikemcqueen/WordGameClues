#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
    node clues -t $line --reject >> reject.out
done < "$1"
