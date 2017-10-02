#!/bin/bash
if [ $# -lt 2  ]
then
   echo 'usage: update_all.sh <clue-type> <clue-count>'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1   #clue count
shift
echo "clues: $_ct, count: $_cc"

name=sugar
#_note=p3s.c2-$_cc.x2.$name
#_production=--production
#_save=--save
#_force=--force

#update-1
if [ $_note ]
then
    node note -$_ct --update=$_note $1 $2 $_force #--save
    if [ $? -ne 0 ]
    then
	echo "update failed for $_note"
	exit -1
    fi
    # need to chop off tailing word of note name here, that's our name (e.g. sugar)
else
    node note -$_ct --update=$_note $1 $2 $_force $_save
    if [ $? -ne 0 ]
    then
	echo "update all failed"
	exit -1
    fi

    echo "Generating new clues.."
    node ../clues -$_ct -c2,$_cc -x2 > tmp/$_ct.c2-$_cc.x2

    #for each name in  note (or file ../../data/words/$_ct.txt
    _input="../../data/words/$_ct.txt"
    echo "update all from $_input"
    while read -r _name
    do
	if [ ! -z $_name ]
	then
	    echo "Grepping..."
	    grep $_name tmp/$_ct.c2-$_cc.x2 > tmp/$_ct.c2-$_cc.x2.$_name

	    echo "Filtering..."
	    _note=$_ct.c2-$_cc.x2.$_name
	    node filter -$_ct tmp/$_note > tmp/$_note.filtered
	    if [ $? -ne 0 ]
	    then
		echo "filter failed on $_note"
		exit -1
	    fi
	    echo "Merging..."
	    #does merge-note work if note doesn't exist? probably not. should it? easier that way.
	    #how about --create flag? otherwise i need note get <note> --quiet to return 0/1 then note create <note>.
	    node note-merge -$_ct tmp/$_note.filtered --note $_note $1 $2
	fi
    done < "$_input"
    exit -1
fi



#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors
