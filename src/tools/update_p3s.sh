#!/bin/bash
if [ $# -eq 0  ]
then
   echo argument required
   exit -1
fi
echo $1

#_production=--production
#_save=--save
_ct=p3s    #clue type

#make update_p3s.sh read words file.
#all console.log(err) -> console.error(err)
#note update -p3s (update all)
#does merge-note work if note doesn't exist? probably not. should it? easier that way.
#how about --create flag? otherwise i need note get <note> --quiet to return 0/1 then note create <note>.

#update-1
node note -$_ct --update=$1 $2 $3 $4
#update-all:
#node note -$_ct --update --production

#for each name in  note (or file ../../data/words/$_ct.txt
name=sugar

if [ !save ]
then
    echo 'Generating new clues..'
    node ../clues -$_ct -c2,6 -x2 > tmp/$_ct.c2-6.x2

    grep $name tmp/$_ct.c2-6.x2 > tmp/$_ct.c2-6.x2.$name

    echo 'Filtering...'
    node filter -$_ct   tmp/$_ct.c2-6.x2.$name > tmp/$_ct.c2-6.x2.$name.filtered
fi

node note-merge tmp/$_ct.c2-6.x2.$name.filtered --note $1 $2 $3 $4

#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors

