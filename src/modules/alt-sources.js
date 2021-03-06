//
// alt-sources.js
//
    
'use strict';

// export a singleton

module.exports = exports = new AltSources();

//

const _                     = require('lodash');
const ClueManager           = require('./clue-manager');
const ComboSearch           = require('./combo-search');
const Debug                 = require('debug')('alt-sources');
const Duration              = require('duration');
const NameCount             = require('../types/name-count');
const PrettyMs              = require('pretty-ms');
const Validator             = require('./validator');

//

const FIRST_COLUMN_WIDTH    = 15;
const SECOND_COLUMN_WIDTH   = 25;

// TODO: this doesn't need to have a constructor.  Can be a true module.

// constructor
// 

function AltSources() {
    this.logging = false;
}

//
//

AltSources.prototype.log = function (text) {
    if (this.logging) {
        console.log(text);
    }
}

//
//
//

AltSources.prototype.show = function (args) {
    if (args.all) {
        this.showAllAlternates({
            count: args.count,
            output: args.output
        });
    }
    else {
        this.showAlternates({
            name:   args.name,
            // TODO: count: args.count,
            output: args.output
        });
    }
}

//
//

AltSources.prototype.showAllAlternates = function (args) {
    if (args.output && !args.count) {
        console.log('WARNING: output format ignored, no -c COUNT specified');
    }
    if (!args.output) {
        console.log('showAlternates: all');
    }
    let anyAdded = false;
    let max = ClueManager.maxClues;
    let startTime = new Date();
    let options = { output: args.output };
    let validateDurationMs = 0;
    do {
        for (let index = 2; index <= max; ++index) {
            let map = ClueManager.knownClueMapArray[index];
            _.keys(map).forEach(name => {
                if (this.showAllAlternatesForNc(
                    NameCount.makeNew(name, index), args.count, options)) {

                    anyAdded = true;
                }
                validateDurationMs += options.validateDurationMs;
            });
            if (!args.output) {
                let durationMs = new Duration(startTime, new Date()).milliseconds;
                console.log(`iter ${index}, duration ${PrettyMs(durationMs)}` +
                            `, validate ${PrettyMs(validateDurationMs)}`);
            }
        }
        // of course, args.output is the only way anyAdded gets to be true
        if (anyAdded && !args.output) {
            console.log('//////////////////');
            console.log('//    loop      //');
            console.log('//////////////////');
        }
    } while (anyAdded);
    if (args.output && args.count) {
        console.log(ClueManager.clueListArray[args.count].toJSON());
    }
}

//
// 

AltSources.prototype.showAllAlternatesForNc = function (nc, count, options = {}) {
    let ncLstAryAry;
    let clue;
    let added = false;

    Debug(`${nc.name}:${nc.count}`);
    ncLstAryAry = ComboSearch.findAlternateSourcesForNc(nc, options);
    if (_.isEmpty(ncLstAryAry)) {
        return;
    }

    // TODO: can I make this parallel with map?
    ncLstAryAry.forEach((ncListArray, index) => {
        if (ncListArray.length === 0) {
            return; // forEach.continue
        }
        if (options.output && count) {
            // TODO: comment why we're not checking if (count == index)
            //
            // display only specific index, in JSON. in this case we just
            // add a new clue to the master cluelist, and later we'll
            // display the entire cluelist.
            clue = getAlternateClue(nc.name, ncListArray);
            
            if (this.logging) {
                console.log ('ADDING: name: ' + clue.name + ', src: ' + clue.src);
            }

            added = ClueManager.addClue(index, clue);
        } else if (count) {
            if (count === index) {
                // display only specific index
                displayAlternate(nc.name, index, ncListArray);
            }
        } else {
            // display all
            displayAlternate(nc.name, index, ncListArray);
        }
    });
    return added;
}

//
//

AltSources.prototype.showAlternates = function (args) {
    let argList;
    let count;
    let name;
    let nc;
    let ncLstAryAry;
    let ncLstAry;

    argList = args.name.split(','); // 'name:N,C' -> [ 'name:N', 'C' ]
    if (argList.length > 1) {
        count = argList[1]; // 'C'
    }
    if (args.output && !count) {
        console.log('WARNING: output format ignored, no ",count" specified');
    }

    name = argList[0]; // 'name:N'
    nc = NameCount.makeNew(name);

    if (!nc.count) {
        throw new Error('Need to supply a count as name:count (for now)');
    }
    if (!args.output) {
        console.log('showAlternates: ' + nc);
    }
    let options = { count };
    ncLstAryAry = ComboSearch.findAlternateSourcesForNc(name, options);
    if (!ncLstAryAry.length) {
        if (!args.output) {
            console.log('No alternates found.');
        }
        return;
    }

    if (this.logging) {
        this.log(name + ' : ' + count + ' : ' + args.output);
    }

    if (args.output && count) {
        displayModifiedClueList(count, getAlternateClue(name, ncLstAryAry[count]));
    }
    else if (count) {
        ncLstAry = ncLstAryAry[count];
        displayAlternate(name, count, ncLstAry);
    }
    else {
        ncLstAryAry.forEach((ncLstAry, index) => {
            displayAlternate(name, index, ncLstAry);
        });
    }
}

//

function displayAlternate (name, count, ncListArray) {
    let s;
    let nameList;
    let found = false;

    s = name + '[' + count + '] ';
    s += format2(s, 20) + ' ';
    ncListArray.forEach((ncList, nclaIndex) => {
        nameList = ncList.map(nc => nc.name);
        nameList.sort(); // really necessary?
        if (ClueManager.knownSourceMapArray[count][nameList.toString()]) {
            //console.log('found: ' + nameList + ' in ' + count);
            return; // continue
        }

        if (found) {
            s += ', ';
        }
        ncList.forEach((nc, nclIndex) => {
            if (nclIndex > 0) {
                s += ' ';
            }
            s += nc;
        });
        found = true;
    });
    if (found) {
        console.log(s);
    }
}

//

function format2 (text, span)
{
    let result = "";
    for (let len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//

function displayModifiedClueList (count, clue) {
    let clueList = ClueManager.clueListArray[count];
    
    clueList.push(clue);
    
    console.log(clueList.toJSON());
}

//

function getAlternateClue (name,  ncListArray) {
    let srcList;
    srcList = [];
    // no loop here because entries will always have the
    // same sources, so just add the first one
    ncListArray[0].forEach(nc => {
        srcList.push(nc.name);
    });
    return {
        name: name,
        src:  _.toString(srcList)
    };
}

