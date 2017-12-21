#!/bin/bash
if [ $# -lt 2  ]
then
   echo 'usage: update_all.sh <clue-type> <clue-count>  [note-name] [--dry-run] [--production] [--match chars] [--from-fs] [--from-en]'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1  #clue count
shift

_name=""
_options=""
while [[ $# -gt 0 ]]
do
      if [[ $1 == '--production' ]]
      then
          echo "--PRODUCTION"
          _production=$1
          _options="$_options --save"
      elif [[ $1 == '--dry-run' ]]
      then
          echo "--DRY RUN"
          _options="$_options $1"
      elif [[ $1 == '--verbose' ]]
      then
          echo "--VERBOSE"
          _options="$_options $1"
      elif [[ $1 == '--from-fs' ]]
      then
          echo "Updating from filesystem..."
          _from_fs=true
      elif [[ $1 == '--from-en' ]]
      then
          echo "Updating from Evernote..."
          _from_en=true
      elif [[ $1 == --match* ]]
      then
          shift
          if [[ $# -eq 0 ]]
          then
              echo "--match requires an argument"
              exit -1
          fi
          echo "--MATCH: $1"
          _match=$1
      elif [[ -z $_name ]]
      then
          _name=$1
      else
          echo "multiple note names or unsupported option: 1) $_name 2) $1"
          exit -1
      fi
      shift
done

echo "clues: $_ct, count: $_cc, note: $_note"

_base=$_ct.c2-$_cc.x2

_out=tmp/update_all.err
echo $(date) >> $_out

#_force=--force

update () {
    if [[ ! -z $_name ]]
    then
        #
        #  update one note
        #
        _note=$_base.$_name
        node note -$_ct --update=$_note $1 $2 $_production $_options 2>> $_out
        _exitcode=$?
        if [[ $_exitcode -ne 0 ]]
        then
            echo "$_exitcode updated"
            #exit -1
        fi
    else
        #
        # update all notes
        #
        node note -$_ct --match $_ct.c2-$_cc.x2.$_match $1 $2 $_production $_options --update 2>> $_out
        _exitcode=$?
        if [[ $_exitcode -ne 0 ]]
        then
            echo "update all failed, $_exitcode"
            exit -1
        fi
    fi
    return $_exitcode
}

if [[ ! -z $_from_en  ]]
then
    update
fi
if [[ ! -z $_from_fs  ]]
then
    _options="$_options --from-fs"
    update
fi

exit 0

echo "Generating new clues.."
node ../clues -$_ct -c2,$_cc -x2 > tmp/$_ct.c2-$_cc.x2 2>> $_out

if [[ ! -z $_name ]]
then
    ./filtermerge.sh $_ct $_cc $_name $_production 
else
    #for each name in  ../../data/words/$_ct.txt
    _wordsfile="../../data/words/$_ct.txt"
    echo "update all from $_wordsfile"
    while read -r _word
    do
        if [ ! -z $_word ]
        then
            ./filtermerge.sh $_ct $_cc $_word $_production 
        fi
    done < "$_wordsfile"
fi
    
#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors
