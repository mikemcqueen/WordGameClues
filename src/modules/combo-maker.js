//
// combo-maker.js
//

'use strict';

// export a singleton

module.exports = exports = new ComboMaker();

//

const _           = require('lodash');
const ClueManager = require('./clue-manager');
const ClueList    = require('../types/clue-list');
const Debug       = require('debug')('combo-maker');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Log         = require('./log')('combo-maker');
const NameCount   = require('../types/name-count');
const PrettyMs    = require('pretty-ms');
const ResultMap   = require('../types/result-map');
const Stringify   = require('stringify-object');
const Validator   = require('./validator');
//const Peco        = require('./peco'); // use this at some point

let logging = false;


//
//

function ComboMaker() {
    this.hash = {};
}

//

ComboMaker.prototype.matchAny = function (srcList, nameList) {
    for (const source of srcList) {
        for (const name of nameList) {
            const regex = new RegExp(`${name}`);
            if (source.match(regex)) return true;
        }
    }
    return false;
};

function getPrimaryNameSrcLists (entries) {
    const nameSrcLists = [];
    entries.forEach(entry => {
        entry.results.forEach(result => {
            nameSrcLists.push(result.ncList.map(nc => ClueManager.primaryNcToNameSrc(nc)));
        });
    });
    return nameSrcLists;
}

function getCompatiblePrimaryNameSrcLists (nameSrcLists1, nameSrcLists2) {
    let nameSrcLists = [];
    for (const nameSrcList1 of nameSrcLists1) {
        for (const nameSrcList2 of nameSrcLists2) {
            Debug(`nameSrcList1: ${nameSrcList1}, nameSrcList2: ${nameSrcList2}`);
            let countList1 = NameCount.makeCountList(nameSrcList1);
            let countList2 = NameCount.makeCountList(nameSrcList2);
            if (_.isEmpty(_.intersectionBy(countList1, countList2, _.toNumber))) {
                nameSrcLists.push(_.concat(nameSrcList1, nameSrcList2));
            }
        }
    }
    return nameSrcLists;
}

function getSourcesList (srcMapEntries) {
    const sources = [];
    srcMapEntries.forEach(entry => {
        entry.results.forEach(result => {
            result.primaryNameSrcList = result.ncList.map(nc => ClueManager.primaryNcToNameSrc(nc));
            result.primarySrcList = NameCount.makeCountList(result.primaryNameSrcList);
	    result.srcNcLists = result.resultMap ? buildSrcNcLists(result.resultMap.map()) : [ result.ncList ];
	    if (logging) {
		console.log(`result ncList ${result.ncList}, srcNcLists ${showNcLists(result.srcNcLists)}`);
		if (_.isEmpty(result.srcNcLists)) console.log(`empty srcNcList: ${Stringify(result.resultMap.map())}`);
	    }
            sources.push(result);
        });
    });
    return sources;
}

function showNcLists (ncLists) {
    let str = "";
    let first = true;
    for (let ncList of ncLists) {
	if (!first) str += ' - ';
	str += ncList;
	first = false;
    }
    return _.isEmpty(str) ? "[]" : str;
}

function mergeSources (sources1, sources2) {
    let mergedSources = {};
    mergedSources.ncList = _.concat(sources1.ncList, sources2.ncList);
    mergedSources.primaryNameSrcList = _.concat(sources1.primaryNameSrcList, sources2.primaryNameSrcList);
    mergedSources.primarySrcList = NameCount.makeCountList(mergedSources.primaryNameSrcList);
    // move to getSourcesLists
    let srcNcLists1 = sources1.srcNcLists;
    let srcNcLists2 = sources2.srcNcLists;
    if (logging) console.log(`srcNcLists1: ${showNcLists(srcNcLists1)}`);
    if (logging) console.log(`srcNcLists2: ${showNcLists(srcNcLists2)}`);
    mergedSources.srcNcLists = _.concat(srcNcLists1, srcNcLists2);
    if (logging) console.log(`  merged: ${showNcLists(mergedSources.srcNcLists)}`);

    return mergedSources;
}

function mergeCompatibleSourcesLists (sources1, sources2) { // TODO sourcesList1, sourcesList2
    let mergedSources = []; // TODO mergedSourcesList
    for (const entry1 of sources1) { // TODO sources1 of sourcesList1
        for (const entry2 of sources2) { // TODO sources2 of sourcesList2
            Debug(`nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
            if (logging) console.log(`mergeCompat: nameSrcList1: ${entry1.primaryNameSrcList}, nameSrcList2: ${entry2.primaryNameSrcList}`);
            if (_.isEmpty(_.intersectionBy(entry1.primarySrcList, entry2.primarySrcList, _.toNumber))) {
                mergedSources.push(mergeSources(entry1, entry2));
            }
        }
    }
    return mergedSources;
}

function mergeAllCompatibleSources (ncList) {
    Expect(ncList.length).is.above(0);
    let sources = getSourcesList(ClueManager.getKnownSourceMapEntries(ncList[0]));
    for (let ncIndex = 1; ncIndex < ncList.length; ncIndex += 1) {
        const nextSources = getSourcesList(ClueManager.getKnownSourceMapEntries(ncList[ncIndex]));
        sources = mergeCompatibleSourcesLists(sources, nextSources);
        if (_.isEmpty(sources)) break;
    }
    return sources;
}

//{
//  'jack:3': {
//    'card:2': {
//      'bird:1,red:1': [   // multiple primary sources with array value type, split them
//        'bird:2,red:8'
//      ]
//    },
//    'face:1': {
//      'face:1': [         // single primary source with array value type, ignore
//        'face:10'
//      ]
//    }
//  }
//}
//
//{
//  'face:1': [              // single top-level primary source with array value type, allow
//    'face:10'
//  ]
//}


function recursiveAddSrcNcLists (list, obj, top) {
    // TODO This is broken for top-level primary sources as above

    let keys = _.flatMap(_.keys(obj), key => {
        let val = obj[key];
        if (_.isObject(val)) {
            if (!_.isArray(val)) return key;
            // split multiple primary sources into separate keys
            let multiplePrimarySourceKeys = key.split(',');
            if (multiplePrimarySourceKeys.length > 1) return multiplePrimarySourceKeys;
	    // allow top level single primary source key with array value type
	    if (top) return key;
            // ignore nested single primary source key with array value type
        }
        return [];
    });
    if (!_.isEmpty(keys)) {
        list.push(keys);
        keys.forEach(key => {
            if (obj[key]) recursiveAddSrcNcLists(list, obj[key], false);
        });
    }
    return list;
}

function buildSrcNcLists (obj) {
    return recursiveAddSrcNcLists([], obj, true);
}

function partialMatchAnyNcList (ncList, matchNcLists) {
    for (let matchNcList of matchNcLists) {
	let match = _.intersectionBy(ncList, matchNcList, _.toString).length == ncList.length;
	// log nclist, matchlist, match
	if (match) return true;
    }
    return false;
}


function mergeAllUsedSources (sourcesList, useNcLists) {
    for (let useNcList of useNcLists) {
        let mergedSourcesList = [];
        let useSourcesList = mergeAllCompatibleSources(useNcList);
        if (_.isEmpty(useSourcesList)) throw new Error(`sources not compatible: ${useNcList}`);
        for (let useSources of useSourcesList) {
            for (let sources of sourcesList) {
                const numCommonPrimarySources = _.intersectionBy(sources.primarySrcList, useSources.primarySrcList, _.toNumber).length;
                let valid = numCommonPrimarySources === 0;

                const allCommonPrimarySources = numCommonPrimarySources === useSources.primarySrcList.length;
                const singlePrimaryNc = useNcList.length === 1 && useNcList[0].count === 1;

                if (allCommonPrimarySources) {
                    if (singlePrimaryNc || partialMatchAnyNcList(useNcList, sources.srcNcLists)) {
                        valid = true;
                    }
                }
                if (valid) {
		    // TODO: i get the feeling that is merging ncList is not working here, doubling up ncList when merging face,card
                    mergedSourcesList.push(mergeSources(sources, useSources));
                }
                if (logging) {
		    console.log(`  valid: ${valid}, useNcList: ${useNcList}`);
		    console.log(`    sources:   ${showNcLists(sources.srcNcLists)}, primary: ${sources.primaryNameSrcList}`);
		    console.log(`    useNcList: ${useNcList}, primary: ${useSources.primaryNameSrcList}`);
		    console.log(`    allCommon: ${allCommonPrimarySources}, singlePrimaryNc: ${singlePrimaryNc}`);
		}
            }
        }
        sourcesList = mergedSourcesList;
    }
    return sourcesList;
}

//
// args:
//  count:   # of primary clues to combine
//  max:     max # of sources to use
//  require: required clue counts, e.g. [3,5,8]
//  sources: limit to these primary sources, e.g. [1,9,14]
//  use:     list of clue name-counts, e.g. ['john:1','bob:5']
//
// A "clueSourceList" is a list (array) where each element is a
// object that contains a list (cluelist) and a count, such as
// [ { list:clues1, count:1 },{ list:clues2, count:2 }].
//
ComboMaker.prototype.makeCombos = function(args, options = {}) {
    let successDuration = 0;
    let failDuration = 0;
    let nextDuration = 0;

    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
    }

    // TODO USE "validateArgs" 

    let require = args.require ? _.clone(args.require) : [];
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');
    if (options.remaining) throw new Error('remaining not yet supported');

    let useNcLists;
    if (args.use) {
        useNcLists = this.buildUseNcLists(args.use);
        Debug(`useNcLists: ${useNcLists}`);
    }

    this.hash = {};
    let csvNameList = [];
    let totalCount = 0;
    let skipCount = 0;
    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({
        sum:     args.sum,
        max:     args.max
        //,require: require
    }).forEach(clueSourceList => {
        Debug(`clueSourceList: ${Stringify(clueSourceList)}`);
        let sourceIndexes = [];

        let result = this.first(clueSourceList, sourceIndexes);
        if (result.done) throw new Error(`no valid combos sources`);

        // this is effectively Peco.getCombinations().forEach()
        let first = true;
        while (!result.done) {
            if (!first) {
                result = this.next(clueSourceList, sourceIndexes, options);
                if (result.done) break;
            } else {
                first = false;
            }

            Log.info(`result.nameList: ${result.nameList}`);
            Log.info(`result.ncList: ${result.ncList}`);

            //logging = result.nameList.toString() === 'king,pitcher' ;
	    //		|| result.nameList.toString() === 'cardinal,smith';

            let sources = mergeAllCompatibleSources(result.ncList);
            
            if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sources)}`);

            // failed to find any compatible combos
            if (_.isEmpty(sources)) continue;

            // TODO maybe: save each sources.origNcList here

            if (useNcLists) {
                sources = mergeAllUsedSources(sources, useNcLists);
            }
            
            if (logging) console.log(`  compatible with used clues: ${!_.isEmpty(sources)}`);

            if (_.isEmpty(sources)) continue;

            if (csvNameList.length < args.maxResults) {
                csvNameList.push(result.nameList.toString());
            }
            if ((++totalCount % 10000) === 0) {
                Debug(`total(${totalCount}), hash(${_.size(this.hash)}), list(${csvNameList.length})`);
            }
        }
    }, this);

    Debug(`success: ${successDuration}ms` +
          `, fail: ${failDuration}ms` +
          `, next: ${nextDuration}ms`);
    Debug(`total(${totalCount})` +
          `, dupeClue(${this.nextDupeClue})` +
          `, dupeSrc(${this.nextDupeSrc})` +
          `, dupeCombo(${this.nextDupeCombo})` +
          `, skip(${skipCount})`);

    return csvNameList;
};

//
// args:
//  count:   # of primary clues to combine
//  max:     max # of sources to use
//  require: required clue counts, e.g. [3,5,8]
//  sources: limit to these primary sources, e.g. [1,9,14]
//  use:     list of clue name-counts, e.g. ['john:1','bob:5']
//
// A "clueSourceList" is a list (array) where each element is a
// object that contains a list (cluelist) and a count, such as
// [ { list:clues1, count:1 },{ list:clues2, count:2 }].
//
ComboMaker.prototype.old_makeCombos = function(args, options = {}) {
    let successDuration = 0;
    let failDuration = 0;
    let nextDuration = 0;

    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
    }

    // TODO USE "validateArgs" 

    let require = args.require ? _.clone(args.require) : [];
    let useNcList;
    let useNameList;
    let useSum = 0;
    if (args.use) {
        let buildResult = this.old_buildUseNcList(args.use);
        useNcList = buildResult.ncList;
        useNameList = NameCount.makeNameList(useNcList);
        useSum = NameCount.makeCountList(useNcList).reduce((a, b) => (a + b));
        Debug(`useNcList: ${useNcList}, useNameList: ${useNameList}, sum: ${useSum}`);
    }
    let validateAll = false;
    if (args.sources) {
        Debug('Validating sources: ' + args.sources);
        validateAll = true;
    }
    //Debug(`require: ${require}`);

    this.hash = {};
    let csvNameList = [];
    let totalCount = 0;
    let skipCount = 0;
    // for each sourceList in sourceListArray
    ClueManager.getClueSourceListArray({
        sum:     args.sum,
        max:     args.max,
        require: require
    }).forEach(clueSourceList => {
        //Debug(`clueSourceList: ${Stringify(clueSourceList)}`);
        let sourceIndexes = [];

        let result = this.first(clueSourceList, sourceIndexes);
        if (result.done) {
            // this shouldn't normally fire, but has happened with
            // 3,3 (count:6) when only 1 unique source in clues3.json
            throw new Error(`no valid combos sources`);
        }

        // this is effectively Peco.getCombinations().forEach()
        let first = true;
        while (!result.done) {
            if (!first) {
                result = this.next(clueSourceList, sourceIndexes, options);
                if (result.done) break;
            } else {
                first = false;
            }

            //Log.info(`result.nameList: ${result.nameList}`);
            //Log.info(`result.ncList: ${result.ncList}`);

            /*
            // build list of clue names from list of clue sources and sourceIndex array
            let clueNameList = clueSourceList.map(
                (clueSource, index) => clueSource.list[sourceIndexes[index]].name);

            // DUBIOUS! filter out clue lists with duplicate clue names.
            if (_.uniq(clueNameList).length !== clueNameList.length) {
                Expect(true).is.false(); // because we filter these out in next()
                continue;
            }
            */

            // if useNcList, all nc must exist in current combo's nc list
            // NEW WAY - is this true? no. not every combo must contain a used NC,
            // combo's must only be compatible with all used NC.s
            let useCountList = [];
            if (useNcList) {
                if (options.allow_used) {
                    // TODO: filter NCs out of useNcList that are in result.ncList
                    const useNcDiff = _.differenceBy(useNcList, result.ncList, _.toString);
                    useNameList = NameCount.makeNameList(useNcDiff);
                    useCountList = NameCount.makeCountList(useNcDiff);
                    useSum = useCountList.reduce((a, b) => (a + b), 0);
                } else {
                    const numUsed = _.intersectionBy(useNcList, result.ncList, _.toString).length;
                    // it is entirely possible to have a combo that doesn't
                    // have any values from useNcList, it only has to be
                    // *compatible* with useNcList.
                    /*
                      if (numUsed !== useNcList.length) {
                      Debug(`skip1: ${result.ncList}`);
                      ++skipCount;
                      continue;
                      }
                    */
                    /* I don't want to skip used.  --option maybe.
                    if (numUsed > 0) {
                        Debug(`skip used (${numUsed}): ${result.ncList}`);
                        continue;
                    }
                    */
                }
            }

            // if --remaining was specified, filter out all source lists
            // where any source matches a named note 'name' suffix
            if (options.remaining && this.matchAny(result.nameList, options.note_names)) {
                continue;
            }

            // NEW WAY - add all used clues to nameList that are not already in it,
            //           and increase sum accordingly
            let nameList = result.nameList;
            let sum = args.sum;
            if (useNameList) {
                nameList = _.concat(nameList, useNameList);
                sum += useSum;
            }

            // NEW WAY - add all used clue counts to 'require', might speed it up a bit
            let requiredAndUsedCounts = _.concat(require, useCountList);

//            let start = new Date();
            let validateResult = { success: true };
            if (!options.merge_style) {
//              console.log(`validating: ${nameList}, sum ${sum}, useNameList ${useNameList}`);
                validateResult = Validator.validateSources({
                    sum/*:         args.sum*/,
                    nameList/*:    result.nameList*/,
                    count:       /*result.*/nameList.length,
                    require:     requiredAndUsedCounts,
                    validateAll: validateAll
                });
            }
//            let duration = new Duration(start, new Date());

//          console.log(`valid: ${validateResult.success}, duration: ${PrettyMs(duration.milliseconds)}`);

            if (validateResult.success) {

//                successDuration += duration.milliseconds;

                if (validateAll) {
                    if (!this.checkPrimarySources(validateResult.list, args.sources)) {
                        Log.debug(`checkPrimarySources.fail`);
                        continue;
                    }
                }
                // if output as primary clues
                if (options.primary) {
                    validateResult.list.forEach(vr => {
                        Debug(`${vr.ncList}`);
                        if (csvNameList.length < args.maxResults) {
                            const nameList = NameCount.makeNameList(vr.ncList).sort();
                            csvNameList.push(nameList.toString());
                        }
                    });
                } else if (csvNameList.length < args.maxResults) {
                    csvNameList.push(result.nameList.toString());
                }
                if ((++totalCount % 10000) === 0) {
                    Debug(`total(${totalCount}), hash(${_.size(this.hash)}), list(${csvNameList.length})`);
                }
            }
            else {
                //Log.info("validateResult.fail");
            }
        }
    }, this);

    Debug(`success: ${successDuration}ms` +
          `, fail: ${failDuration}ms` +
          `, next: ${nextDuration}ms`);
    Debug(`total(${totalCount})` +
          `, dupeClue(${this.nextDupeClue})` +
          `, dupeSrc(${this.nextDupeSrc})` +
          `, dupeCombo(${this.nextDupeCombo})` +
          `, skip(${skipCount})`);

    return csvNameList;
};

// As long as one final result has only primary sources from 'sources'
// array, we're good.

ComboMaker.prototype.checkPrimarySources = function(resultList, sources) {
    return resultList.some(result => {
        return NameCount.makeCountList(result.nameSrcList)
            .every(source => {
                return _.includes(sources, source);
            });
    });
};

//

ComboMaker.prototype.buildUseNcLists = function(useArgsList) {
    let useNcLists = [];
    useArgsList.forEach(useArg =>  {
        let args = useArg.split(',');
        let ncList = args.map(arg => {
            let nc = NameCount.makeNew(arg);
            if (!nc.count) throw new Error(`arg: ${arg} requires a :COUNT`);
            if (!_.has(ClueManager.knownClueMapArray[nc.count], nc.name)) throw new Error(`arg: ${nc} does not exist`);
            return nc;
        });
        useNcLists.push(ncList);
    });
    return useNcLists;
};

ComboMaker.prototype.old_buildUseNcList = function(args) {
    let ncList = [];
    let countList = [];
    args.forEach(arg =>  {
        let nc = NameCount.makeNew(arg);
        if (!nc.count) throw new Error(`arg: ${arg} requires a :COUNT`);
        if (!_.has(ClueManager.knownClueMapArray[nc.count], nc.name)) throw new Error(`arg: ${nc} does not exist`);
        countList.push(nc.count);
        ncList.push(nc);
    });
    return { ncList, countList };
};

//
//
ComboMaker.prototype.hasUniqueClues = function(clueList) {
    let sourceMap = {};
    for (let clue of clueList) {
        if (isNaN(clue.count)) {
            throw new Error('bad clue count');
        }
        else if (clue.count > 1) {
            // nothing?
        }
        else if (!this.testSetKey(sourceMap, clue.src)) {
            return false; // forEach.continue... ..why?
        }
    }
    return true;
};

//

ComboMaker.prototype.testSetKey = function(map, key, value = true) {
    if (_.has(map, key)) return false;
    map[key] = value;
    return true;
};

//

ComboMaker.prototype.displaySourceListArray = function(sourceListArray) {
    console.log('-----\n');
    sourceListArray.forEach(function(sourceList) {
        sourceList.forEach(function(source) {
            source.display();
            console.log('');
        });
        console.log('-----\n');
    });
};

//

ComboMaker.prototype.first = function(clueSourceList, sourceIndexes, options = {}) {
    for (let index = 0; index < clueSourceList.length; ++index) {
        sourceIndexes[index] = 0;
    }
    sourceIndexes[sourceIndexes.length - 1] = -1;
    return this.next(clueSourceList, sourceIndexes, options);
};

//

ComboMaker.prototype.next = function(clueSourceList, sourceIndexes, options = {}) {
    for (;;) {
        if (!this.nextIndex(clueSourceList, sourceIndexes, options)) {
            return { done: true };
        }
        let ncList = [];          // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        let nameList = [];        // e.g. [ "pollock", "jackson" ]
        let srcCountStrList = []; // e.g. [ "white,fish:2", "moon,walker:4" ]
        if (!clueSourceList.every((clueSource, index) => {
            let clue = clueSource.list[sourceIndexes[index]];
            if (clue.ignore || clue.skip) {
                return false; // every.exit
            }
            nameList.push(clue.name);
            // I think this is right
            ncList.push(NameCount.makeNew(clue.name, clueSource.count));
            srcCountStrList.push(NameCount.makeCanonicalName(clue.src, clueSource.count));
            return true; // every.continue;
        })) {
            continue;
        }

        nameList.sort();
        // skip combinations we've already checked
        if (!this.addComboToFoundHash(nameList.toString())) continue; // already checked

        // skip combinations that have duplicate source:count
        if (!options.allow_dupe_src) {
            if (_.uniq(srcCountStrList).length !== srcCountStrList.length) {
                //Debug('skipping duplicate clue src: ' + srcCountStrList);
                ++this.nextDupeSrc;
                continue;
            }
        }

        // skip combinations that have duplicate names
        if (_.sortedUniq(nameList).length !== nameList.length) {
            //Debug('skipping duplicate clue name: ' + nameList);
            ++this.nextDupeClue; // TODO: DupeName
            continue;
        }

        return {
            done:     false,
            ncList:   ncList.sort(),
            nameList: nameList
        };
    }
};

//
//
ComboMaker.prototype.addComboToFoundHash = function(nameListCsv) {
    if (this.testSetKey(this.hash, nameListCsv)) {
        this.hash[nameListCsv] = true;
        return true;
    }
    //Debug('skipping duplicate combo: ' + nameListCsv);
    ++this.nextDupeCombo;
    return false;
};

//
//
ComboMaker.prototype.nextIndex = function(clueSourceList, sourceIndexes) {
    let index = sourceIndexes.length - 1;

    // increment last index
    ++sourceIndexes[index];

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    while (sourceIndexes[index] === clueSourceList[index].list.length) {
        sourceIndexes[index] = 0;
        --index;
        if (index < 0) {
            return false;
        }
        ++sourceIndexes[index];
    }
    return true;
};

//
//
ComboMaker.prototype.displayCombos = function(clueListArray) {
    console.log('\n-----\n');
    let count = 0;
    clueListArray.forEach(function(clueList) {
        clueList.display();
        ++count;
    });
    console.log('total = ' + count);
};

//
//
ComboMaker.prototype.clueListToString = function(clueList) {
    let str = '';
    clueList.forEach(function(clue) {
        if (str.length > 0) {
            str += ' ';
        }
        str += clue.name;
        if (clue.src) {
            str += ':' + clue.src;
        }
    });
    return str;
};

