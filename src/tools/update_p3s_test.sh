#!/bin/sh
if [ $# -eq 0  ]
then
   echo argument required
   exit -1
fi
echo $1

node note -p3s --update=$1
#update-all:

#clues -c2,6 -x2 > tmp/(p3s).c2-6.x2
#for each name in  note (p3s).wordlist
#--grep name tmp/(p3s).c2-6.x2 > tmp/p3s.c2-6.x2.name
#--filter (-p3s) tmp/(p3s).c2-6.x2 > tmp/p3s.c2-6.x2.name.filtered
#--note-merge tmp/p3s.c2-6.x2.name.filtered (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors

