//
// combo-search.js
//

'use strict';

// export a singleton

module.exports = exports = new ComboSearch();


const _              = require('lodash');

const ClueManager    = require('../dist/modules/clue-manager');
const Validator      = require('../dist/modules/validator');
const ClueList       = require('../dist/types/clue-list');
const NameCount      = require('../dist/types/name-count');

const Peco           = require('./peco');
const Debug          = require('debug')('combo-maker');
const Duration       = require('duration');

//

function ComboSearch() {
    this.logging = true;
    this.logLevel = 0;
}

//

ComboSearch.prototype.log = function(text) {
    let pad;
    let index;
    if (this.logging) {
        pad = '';
        for (let index=0; index<this.logLevel; ++index) {
            pad += ' ';
        }
        console.log(pad + text);
    }
}

//
//

ComboSearch.prototype.findAlternateSourcesForNc = function(nc, options = {}) {
    let srcNameListArray;
    let resultNcListArray = [];
    let peco;

    options.validateDurationMs = 0;

    srcNameListArray = ClueManager.makeSrcNameListArray(nc);
    srcNameListArray.forEach(srcNameList => {
        let curCount;
        let maxCount;
        let countListArray;
        let matchCountListArray = [];

        if (this.logging) {
            this.log('looking for source list ' + srcNameList);
        }
        countListArray = srcNameList.map(name => ClueManager.getCountListForName(name));
        if (this.logging) {
            this.log('count list:');
            countListArray.forEach(countList => {
                this.log(' ' + countList);
            });
        }
        if (options.count) {
            curCount = maxCount = options.count;
        } else {
            curCount = srcNameList.length;
            maxCount = ClueManager.maxClues;
        }
        for (; curCount <= maxCount; ++curCount) {
            if (curCount == nc.count) {
                continue;
            }
            if (this.logging) {
                this.log('  for count ' + curCount);
            }
            Peco.makeNew({
                sum:   curCount,
                count: srcNameList.length
            }).getPermutations().forEach(countList => {
                if (this.logging) {
                    this.log('   in ' + countList);
                }
                if (this.findCountListInCountListArray(countList, countListArray)) {
                    if (!matchCountListArray[curCount]) {
                        matchCountListArray[curCount] = [ countList ];
                    } else {
                        matchCountListArray[curCount].push(countList);
                    }
                    if (this.logging) {
                        this.log('    found! length=' + countList.length);
                    }
                }
                else {
                    if (this.logging) {
                        this.log('    failed');
                    }
                }
            }, this);
        }

        // really: countListArrayArray
        matchCountListArray.forEach((countListArray, claaIndex) => {
            let ncListArray = [];
            countListArray.forEach((countList, claIndex) => {
                let ncList = [];
                let sum = 0;
                let result;
                countList.forEach((count, clIndex) => {
                    sum += count;
                    ncList.push(NameCount.makeNew(srcNameList[clIndex], count));
                });
                if (sum !== claaIndex ) {
                    throw new Error('something i dont understand here obviously');
                }
                let startTime = new Date();
                result = Validator.validateSources({
                    sum:      sum,
                    nameList: srcNameList,
                    count:    srcNameList.length
                });
                options.validateDurationMs += new Duration(startTime, new Date());
                if (result.success) {
                    ncListArray.push(ncList);
                }
            });
            resultNcListArray[claaIndex] = ncListArray;
        });
    });
    return resultNcListArray;
}

// find [1, 2] in { [1,4],[2,5] }
//

ComboSearch.prototype.findCountListInCountListArray =
    function(countList, countListArray)
{
    let indexLengthList;
    let index;
    let resultCountList;

    if (countList.length != countListArray.length) {
        throw new Error('mismatched lengths');
    }

    indexLengthList = [];
    countListArray.forEach(cl => {
        indexLengthList.push({
            index:  0,
            length: cl.length
        });
    });

    do {
        resultCountList = [];
        if (countList.every((count, index) => {
            if (count != countListArray[index][indexLengthList[index].index]) {
                return false;
            }
            return true; // every.continue
        })) {
            return true; // function.exit
        }
    } while (this.nextIndexLength(indexLengthList));

    return null;
}

//
//

ComboSearch.prototype.findNameListInCountList =
    function(nameList, countList)
{
    let ncList;
    let countListArray = [];
    let count;

    if (nameList.length != countList.length) {
        throw new Error('mismatched list lengths');
    }
    return ncList;
}

//
//TODO: 

ComboSearch.prototype.first =
    function(clueSourceList, sourceIndexes)
{
    let index;

    this.hash = {};
    for (index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;

    return this.next(clueSourceList, sourceIndexes);
}

//
//

ComboSearch.prototype.nextIndexLength =
    function(indexLengthList)
{
    let index = indexLengthList.length - 1;

    // increment last index
    ++indexLengthList[index].index;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    while (indexLengthList[index].index >= indexLengthList[index].length) {
        indexLengthList[index].index = 0;
        --index;
        if (index < 0) {
            return false;
        }
        ++indexLengthList[index].index;
    }
    return true;
}

