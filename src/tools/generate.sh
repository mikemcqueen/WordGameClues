#!/bin/bash

if [[ $# -lt 2  ]]
then
   echo 'usage: generate.sh <clue-type> <clue-count>'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1  #clue count
shift

_base=$_ct.c2-$_cc.x2

echo "Generating word pairs.."
node clues -$_ct -c2,$_cc -x2 $1 $2 $3 $4 $5 $6 | sort > tmp/$_base 
