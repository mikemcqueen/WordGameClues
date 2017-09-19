#!/bin/bash
if [ $# -eq 0  ]
then
   echo argument required
   exit -1
fi
echo $1

production=--production
save=

#update-1
node note -p3s --update=$1 $production 
#update-all:
#node note -p3s --update=$1 --production

#for each name in  note (p3s).wordlist
name=sugar

if [ !save ]
then
    echo 'Generating new clues..'
    node ../clues -p3s -c2,6 -x2 > tmp/p3s.c2-6.x2

    grep $name tmp/p3s.c2-6.x2 > tmp/p3s.c2-6.x2.$name

    echo 'Filtering...'
    #TODO: need to pass a flag here saying 'don't filter known urls'
    node filter -p3s --keep known  tmp/p3s.c2-6.x2.$name > tmp/p3s.c2-6.x2.$name.filtered
fi

node note-merge tmp/p3s.c2-6.x2.$name.filtered  $production --note $1

#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors

