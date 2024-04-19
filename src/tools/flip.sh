#!/bin/bash
if [ $# -ne 1  ]
then
   echo 'usage: flip.sh <pairs-filename>'
   exit -1
fi

awk '{ print $1 " " $2; print $2 " " $1 }' $1 |sort | uniq
