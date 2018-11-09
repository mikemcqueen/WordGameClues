#!/bin/bash
if [ $# -lt 3  ]
then
   echo 'usage: filtermerge.sh <clue-type> <clue-count> <word> [--production] [--generate] [--notebook NAME]'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1  #clue count
shift
_name=$1  #word
echo "Name: $_name"
shift

_base=$_ct # .c2-$_cc.x2
_note=$_ct.$_name # .c2-$_cc.x2.$_name

if [[ $_name == "remaining" ]]
then
    _base=$_base.$_name
    _remaining="--remaining"
elif [[ $_name == "all" ]]
then
    _base=$_base.$_name
    _all=$_name
    _generate=true
else
    _grep=$_name
fi

while [[ $# -gt 0 ]]
do
      if [[ $1 == "--production" ]]
      then
          echo "---PRODUCTION---"
          _production=$1
          _save=--save
      elif [[ $1 == "--generate" ]]
      then
           _generate=true
      elif [[ $1 == "--article" ]]
      then
          if [[ ! -z $_remaining ]]
          then
              echo "$1 not allowed with $_remaining"
              exit -1
          fi
          _article=$1
      elif [[ $1 == "--notebook" ]]
      then
           shift
           # check if $# -eq 0 || $1.startsWith('--')
           _notebook=--notebook=$1
      else
          echo "unknown option, $1"
          exit -1
      fi
      shift
done

if [[ ! -z $_article ]]
then
    _note=$_note.article
    _base=$_base.article
elif [[ ! -z $_all ]]
then
    _note=$_note.title
    _base=$_base.title
fi    

_out=tmp/filtermerge.err
echo $(date) >> $_out

if [[ $_generate || ! -z $_remaining ]]
then
    echo "Generating word pairs for $_base..."
    node clues -$_ct -c2,$_cc -x2 $_remaining $_production > tmp/$_base 2>> $_out
fi

if [[ ! -z $_grep ]]
then
    echo "Grepping..."
    grep $_name tmp/$_base > tmp/$_note
fi

_filtered=$_note.filtered

echo "Filtering..."
node filter -$_ct tmp/$_note $_article > tmp/$_filtered 2>> $_out
_exitcode=$?
if [ $_exitcode -ne 0 ]
then
    echo "filter failed on $_note, $_exitcode"
    exit -1
fi

echo "Merging..."
node note-merge -$_ct tmp/$_filtered --note $_note --force-create $_production $_notebook 2>> $_out
_exitcode=$?
if [ $_exitcode -ne 0 ]
then
    echo "merge failed on $_filtered, $_exitcode"
    exit -1
fi
