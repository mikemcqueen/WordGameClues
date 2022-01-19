//
// validator.js
//

'use strict';

// export a singleton, for no particular reason.

//

//const _           = require('lodash');
import _ from 'lodash';
const ClueList    = require('../types/clue-list');
const ClueManager = require('./clue-manager');
const Debug       = require('debug')('validator');
const Expect      = require('should/as-function');
const NameCount   = require('../types/name-count');
const Peco        = require('./peco');
const ResultMap   = require('../types/result-map');
const Stringify   = require('stringify-object');
const Timing      = require('debug')('timing');

let rvsSuccessSeconds = 0;
let rvsFailDuration  = 0;
    
let allowDupeNameSrc = false;
let allowDupeSrc     = false;
let allowDupeName    = true;

let logLevel         = 0;
    
let count = 0;
let dupe = 0;

    // TODO: these are duplicated in ResultMap
let PRIMARY_KEY      = '__primary';
let SOURCES_KEY      = '__sources';

//
//
interface NameCount {
    name: string;
    count: number;
}
type NCList = NameCount[];

type Result = any;

interface RvsResult {
    success: boolean;
    list: Result[];
}

// TODO: options.xp
const xp = true;

/*
let log = function (text) {
    Debug(`${indent()}${text}`);
}
 */


let spaces = (length: number): string => {
    return ' '.repeat(length);
};

let indent = (): string => {
    return spaces(logLevel);
};

let indentNewline = (): string => {
    return '\n' + indent();
};

//
//
let dumpResultMap = (seq: any, level: number = 0): void => {
    if (typeof seq === 'object') {
        console.log(indent() + spaces(2 * level) + (_.isArray(seq) ? '[' : '{'));
        ++level;

        if (_.isArray(seq)) {
            seq.forEach((elem: any) => {
                if (typeof elem === 'object') {
                    dumpResultMap(elem, level + 1);
                } else {
                    console.log(indent() + spaces(2*level) + elem);
                }
            }, this);
        } else {
            _.forOwn(seq, _.bind((value: any, key: any) => {
                if (typeof value === 'object') {
                    console.log(indent() + spaces(2 * level) + key + ':');
                    dumpResultMap(value, level + 1);
                } else {
                    console.log(indent() + spaces(2 * level) + key + ': ' + value);
                }
            }, this));
        }

        --level;
        console.log(indent() + spaces(2 * level) + (_.isArray(seq) ? ']' : '}'));
    }
};

//

let setAllowDupeFlags = (args: any): void => {
    if (!_.isUndefined(args.allowDupeNameSrc)) {
        allowDupeNameSrc = args.allowDupeNameSrc;
    }
    if (!_.isUndefined(args.allowDupeName)) {
        allowDupeName = args.allowDupeName;
    }
    if (!_.isUndefined(args.allowDupeSrc)) {
        allowDupeSrc = args.allowDupeSrc;
    }
};

// in a word: unnecessary

let uniqueResult = (success: boolean, list: any = undefined): RvsResult => {
    return {
	success,
        list
    };
};

//
//
//  origNcList:
//  ncList:
//  nameSrcList:
//  pendingMap:
//  validateAll:
//  ncNameListPairs:
//
let getCompatibleResults = (args: any): any => {
    // no duplicates, and all clues are primary, success!
    Debug('++allUniquePrimary' +
          `${indentNewline()}  origNcList:  ${args.origNcList}` +
          `${indentNewline()}  ncList:      ${args.ncList}` +
          `${indentNewline()}  nameSrcList: ${args.nameSrcList}`);
    if (xp) {
        Expect(args.origNcList).is.an.Array().and.not.empty();
        Expect(args.ncList).is.an.Array().and.not.empty();
        Expect(args.nameSrcList).is.an.Array().and.not.empty();
    }

    let logit = false;
    let resultList: any[] = [];
    if (logit) {
        Debug(`aUP: adding primary result`);
        Debug(`  ${args.ncList}`);
        Debug(`  -as- ${args.nameSrcList}`);
    }
    addCompatibleResult(resultList, args.nameSrcList, args);
    if (_.isEmpty(resultList) || args.validateAll) {
        cyclePrimaryClueSources({ ncList:args.ncList }).some((nameSrcList: NCList) => {
            // check if nameSrcList is already in result list
            if (hasNameSrcList(resultList, nameSrcList)) {
                if (logit) {
                    Debug(`aUP cycle: already in resultList: ${nameSrcList}`);
                }
                return false;
            }
            if (logit) {
                Debug(`aUP cycle: adding primary result`);
                Debug(`  ${args.ncList}`);
                Debug(`  -as- ${nameSrcList}`);
            }
            addCompatibleResult(resultList, nameSrcList, args);
            return !args.validateAll; // some.exit if !validateAll, else some.continue
        });
    }
    return resultList;
};

//
let addCompatibleResult = (resultList: any, nameSrcList: NCList, args: any): void => {
    if (xp) {
        Expect(resultList).is.an.Array();
        Expect(nameSrcList).is.an.Array();
        Expect(args.origNcList).is.an.Array();
        Expect(args.ncList).is.an.Array();
//        Expect(args.ncNameListPairs).is.an.Array();
    }

    resultList.push({
        ncList:      args.ncList,
        nameSrcList: nameSrcList,
        resultMap:   _.cloneDeep(args.pendingMap).addResult({
            origNcList:      args.origNcList,
            primaryNcList:   args.ncList,
            nameSrcList:     nameSrcList,
//            ncNameListPairs: args.ncNameListPairs
        }).ensureUniquePrimaryLists()
    });
};

// Simplified version of checkUniqueSources, for all-primary clues.
// Check all variations of any duplicate-sourced primary clues.
//
// args:
//  ncList:
//  exclueSrcList:
//
let cyclePrimaryClueSources = (args: any): Result[] => {
    if (xp) Expect(args.ncList).is.an.Array().and.not.empty();

    Debug(`++cyclePrimaryClueSources`);

    // must copy the NameCount objects within the list
    let localNcList = _.cloneDeep(args.ncList);
    let resultList: any[] = [];
    let buildArgs: any = {
        ncList:     args.ncList,   // always pass same unmodified ncList
        allPrimary: true
    };
    let buildResult;
    do {
        // build src name list of any duplicate-sourced primary clues
        if (buildResult) {
            buildArgs.indexMap = buildResult.indexMap;
        }
        buildResult = buildSrcNameList(buildArgs);

        // change local copy of ncList sources to buildResult's sources
        localNcList.forEach((nc: NameCount, index: number) => {
            localNcList[index].count = buildResult.primarySrcNameList[index];
        });

        // TODO: return srcMap in findResult
        let srcMap = {};
        let findResult = findDuplicatePrimarySource({
            ncList: localNcList,
            srcMap: srcMap
        });
        if (findResult.duplicateSrc) {
            continue;
        }
        if (xp) Expect(_.size(localNcList), 'localNcList').is.equal(_.size(srcMap));

        let nameSrcList = getNameSrcList(srcMap);
        Debug(`[srcMap] primary keys: ${nameSrcList}`);
        // all the source clues we just validated are primary clues
        Debug(`cycle: adding result: ` +
            `${indentNewline()} ${args.ncList} = ` +
            `${indentNewline()}   ${nameSrcList}`);
        resultList.push(nameSrcList);
    } while (incrementIndexMap(buildResult.indexMap));

    Debug(`--cyclePrimaryClueSources, size: ${_.size(resultList)}`);
    resultList.forEach((result: any) => {
        Debug(`  list: ${result}`);
    });

    return resultList;
};

// ncList:              nameCountList
//
let findDuplicatePrimaryClue = (args: any): any => {
    let duplicateName;
    let duplicateSrc;
    let duplicateSrcName;

    Debug(`++findDuplicatePrimaryClue, ncList: ${args.ncList}`);

    // look for duplicate primary clue sources, return conflict map
    // also checks for duplicate names
    let findResult = findPrimarySourceConflicts({ ncList: args.ncList });
    duplicateName = findResult.duplicateName;

    if (!_.isEmpty(findResult.conflictSrcMap)) {
        // resolve duplicate primary source conflicts
        let resolveResult = resolvePrimarySourceConflicts({
            srcMap:         findResult.srcMap,
            conflictSrcMap: findResult.conflictSrcMap
        });
        duplicateSrcName = resolveResult.duplicateSrcName;
        duplicateSrc = resolveResult.duplicateSrc;
    }

    // log before possible exception, to provide more info
    Debug(`--findDuplicatePrimaryClue` +
          `, duplicateName: ${duplicateName}` +
          `, duplicateSrcName: ${duplicateSrcName}` +
          `, duplicateSrc: ${duplicateSrc}` +
          `, allPrimary: ${findResult.allPrimary}` +
          `, srcMap.size: ${_.size(findResult.srcMap)}`);

    if (findResult.allPrimary && _.isUndefined(duplicateSrc) &&
        (_.size(findResult.srcMap) != _.size(args.ncList)))
    {
        Debug(`ncList: ${args.ncList}`);
        Debug(`srcMap.keys: ${_.keys(findResult.srcMap)}`);
        throw new Error('srcMap.size != ncList.size');
    }

    return {
        duplicateName:    duplicateName,
        duplicateSrcName: duplicateSrcName,
        duplicateSrc:     duplicateSrc,
        allPrimary:       findResult.allPrimary,
        srcMap:           findResult.srcMap
    };
};


// args:
//  ncList:
//
// result:
//

let findPrimarySourceConflicts = (args: any): any => {
    let duplicateName;

    Debug(`++findPrimarySourceConflicts, ncList: ${args.ncList}`);

    if (xp) Expect(args.ncList, 'args.ncList').is.ok();

    let allPrimary = true;
    let nameMap = {};
    let srcMap = {};
    let conflictSrcMap = {};

    args.ncList.forEach((nc: NameCount) => {
        if (nc.count > 1) {
            Debug(`fPSC: non-primary, ${nc}`);
            allPrimary = false;
            return; // forEach.continue
        }
        Debug(`fPSC: primary, ${nc}`);

        // if name is in nameMap then it's a duplicate
        if (_.has(nameMap, nc.name)) {
            duplicateName = nc.name;
        } else {
            nameMap[nc.name] = true;
        }

        let srcList = ClueManager.knownClueMapArray[1][nc.name];
        //console.log(`srcList for ${nc.name}:1, ${srcList}`);
        // look for an as-yet-unused src for the given clue name
        if (!srcList.some((src: any) => {
            if (!_.has(srcMap, src)) {
                srcMap[src] = nc.name;
                return true; // found; some.exit
            }
            return false; // not found; some.continue
        })) {
            // unused src not found: add to conflict map, resolve later
            if (!_.has(conflictSrcMap, srcList)) {
                conflictSrcMap[srcList] = [];
            }
            conflictSrcMap[srcList].push(nc.name);
        }
    }, this);

    Debug(`--findPrimarySourceConflicts: dupeName: `
          + (duplicateName ? duplicateName : 'none')
          + `, allPrimary: ${allPrimary}`
	  + `, conflicts: ${_.size(conflictSrcMap)}`);

    return {
        duplicateName:  duplicateName,
        allPrimary:     allPrimary,
        srcMap:         srcMap,
        conflictSrcMap: conflictSrcMap
    };
};

// args:
//  srcMap:
//  conflictSrcMap:
//

let resolvePrimarySourceConflicts = (args: any): any => {
    let duplicateSrcName;
    let duplicateSrc;

    if (!args.srcMap || !args.conflictSrcMap) {
        throw new Error('missing args' +
                        ', srcMap:' + args.srcMap +
                        ' conflictSrcMap: ' + args.conflictSrcMap);
    }

    Debug(`++resolvePrimarySourceConflicts`);
    Debug(`  srcMap keys: ${_.keys(args.srcMap)}`);
    Debug(`  conflictSrcMap keys: ${_.keys(args.conflictSrcMap)}`);

    // resolve primary source conflicts
    _.keys(args.conflictSrcMap).every((conflictSrc: any) => {
        let srcList = conflictSrc.split(',');
        let conflictNameList = args.conflictSrcMap[conflictSrc];
        Debug(`Attempting to resolve source conflict at ${conflictSrc}, names: ${conflictNameList}`);

        // if conflictNameList.length > srcList.length then there
        // are more uses of this clue than there are sources for it.
        if (conflictNameList.length > srcList.length) {
            duplicateSrcName = conflictNameList.toString();
            duplicateSrc = conflictSrc;
            return false; // every.exit
        }
        // otherwise we may be able to support the clue count; see
        // if any conflicting clue names can be moved to other sources
        if (!srcList.some((src: any) => {
            // look for alternate unused sources for candidateName
            let candidateName = args.srcMap[src];
            let candidateSrcList = ClueManager.knownClueMapArray[1][candidateName];
            Debug(`Candidate sources for ${candidateName}:${src} are [${candidateSrcList}]`);
            if (candidateSrcList.some((candidateSrc: any) => {
                if (!_.has(args.srcMap, candidateSrc)) {
                    Debug(`Successfully resolved ${conflictSrc} as ${candidateSrc}!`);
                    // success! update srcMap
                    args.srcMap[candidateSrc] = candidateName;
                    // any name will do?!
                    args.srcMap[src] = conflictNameList.pop();
                    if (_.isEmpty(conflictNameList)) {
                        return true; // candidateSrcList.some.exit
                    }
                }
                return false; // candidateSrcList.some.continue
            })) {
                return true; // srcList.some.exit
            }
            return false; // srcList.some.continue
        })) {
            // failed to find an alternate unused source for all conflict names
            duplicateSrcName = _.toString(conflictNameList);
            duplicateSrc = conflictSrc;

            Debug(`cannot resolve conflict, names: ${duplicateSrcName}, src: ${duplicateSrc}`);
            Debug(`used sources, `);
            _.keys(args.srcMap).forEach((key: any) => {
                Debug(`  ${key}: ${args.srcMap[key]}`);
            });
            return false; // conflictSrcMap.keys().every.exit
        }
        return true;
    });
    return {
        duplicateSrcName: duplicateSrcName,
        duplicateSrc:     duplicateSrc
    };
};

// args:
//  ncList:      // ALL PRIMARY clues in name:source format (not name:count)
//  srcMap:
//  nameMap:
//
let findDuplicatePrimarySource = (args: any): any => {
    if (xp) Expect(args.ncList).is.an.Array();

    let duplicateSrcName;
    let duplicateSrc;

    args.ncList.some((nc: NameCount) => {
        let src = nc.count;
        if (_.has(args.srcMap, src)) {
            // duplicate source
            duplicateSrcName = nc.name;
            duplicateSrc = src;
            return true; // some.exit
        }
        // mark source used
        args.srcMap[src] = nc.name;
        return false; // some.continue
    });
    return {
        duplicateSrcName: duplicateSrcName,
        duplicateSrc:     duplicateSrc
    };
}

//
//

let evalFindDuplicateResult = (result: any, logPrefix: string): boolean => {
    let dupeType = '';
    let dupeValue = '';

    if (result.duplicateName || result.duplicateSrc) {
        Debug(`duplicate name: ${result.duplicateName}` +
              `, src: ${result.duplicateSrc}`);
    }
    if (!allowDupeName && result.duplicateName) {
        dupeType = 'name';
        dupeValue = result.duplicateName;
    }
    if (!allowDupeSrc && result.duplicateSrc) {
        if (dupeType.length) {
            dupeType += ', ';
            dupeValue += ', ';
        }
        dupeType += 'source';
        dupeValue += result.duplicateSrcName + '(' + result.duplicateSrc + ')';
    }

    // NOTE: need some extra logic here to support:'
    // * NO dupe src with 2-source clue
    // * NO dupe name with 2-source clue

    if (dupeType.length) {
        Debug(`${logPrefix} duplicate primary ${dupeType}, ${dupeValue}`);
        return false;
    }
    return true;
};

// args:
//  ncList
//  allPrimary:  boolean
//  indexMap:
//
// Divide clues in ncList into known compound and known primary sources.
//
// compoundSrcNameList: compound source clue names
// compoundClueCount:   # of primary clues of which compoundSrcNameList consists
// primaryNcList:       primary source clue NCs
// compoundNcList:      subset of args.ncList which contains only compound NCs
//
// NOTE: is problem below solved now?
// So there is some potential trickiness here for a 100% correct solution.
// If a clue has multiple source combinations, we technically need to build
// a separate clueNameList for each possible combination. if two or more clues
// have multiples source combinations, we need to build all combinations of those
// combinations. for the second case, punt until it happens.
//
let buildSrcNameList = (args): any => {
    Debug(`++buildSrcNameList, ncList(${args.ncList.length})` +
          `${indentNewline()}  ncList: ${args.ncList}` +
          `${indentNewline()}  allPrimary: ${args.allPrimary}`);

    let indexMap = getIndexMap(args.indexMap);
    let allPrimary = true;
    let clueCount = 0;
    let compoundNcList: NCList = [];
    let compoundSrcNameList: any[] = [];
    let compoundNcNameListPairs: any[] = [];
    let primaryNcList: NCList = [];
    let primarySrcNameList: any[] = [];
    let primaryPathList: any[] = []; //  TODO: i had half an idea here
    let resultMap = ResultMap.makeNew();

    args.ncList.forEach((nc: /*NameCount*/ any, ncIndex: number) => {
        let src = args.allPrimary ? 1 : nc.count;
        // i.e. srcNameCsvArray
        let srcList = ClueManager.knownClueMapArray[src][nc.name]; // e.g. [ 'src1,src2,src3', 'src2,src3,src4' ]
        if (!srcList) {
            throw new Error('kind of impossible but missing clue!');
        }

        // only do indexing if all clues are primary, or if this
        // is a compound clue
        let slIndex; // srcListIndex
        if (args.allPrimary || (nc.count > 1)) {
            slIndex = getSrcListIndex(indexMap, nc, srcList);
        } else {
            slIndex = 0;
        }
        Debug(`build: index: ${ncIndex}, source: ${srcList[slIndex]}`);

        let srcNameList = srcList[slIndex].split(',');      // e.g. [ 'src1', 'src2', 'src3' ]
        if (nc.count === 1) {
            primaryNcList.push(nc);
            // short circuit if we're called with allPrimary:true.
            if (args.allPrimary) {
                primarySrcNameList.push(...srcNameList);
                return; // forEach.next
            }
        }
        if (xp) Expect(resultMap.map()[nc]).is.undefined();

        // if nc is a primary clue
        if (nc.count == 1) {
            // add map entry for list of primary name:sources
            if (!_.has(resultMap.map(), PRIMARY_KEY)) {
                resultMap.map()[PRIMARY_KEY] = [];
            }
            resultMap.map()[PRIMARY_KEY].push(`${nc}`); // consider nc.name here instead
            return; // forEach.next;
        }
        
        // nc is a compound clue
        compoundNcNameListPairs.push([nc, _.clone(srcNameList)]);
        let map = resultMap.map()[nc] = {};
        // if sources for this nc are all primary clues
        if (_.size(srcNameList) === nc.count) {
            // build component primary NC list
            let localPrimaryNcList = srcNameList.map(name => NameCount.makeNew(name, 1));
            // add map entry for list of (eventual) primary name:sources
            map[localPrimaryNcList] = [];
            primaryNcList.push(...localPrimaryNcList);
            return; // forEach.next;
        }

        // sources for this nc include a compound clue
        clueCount += nc.count;
        allPrimary = false;

        // add map entry for list of source names
        // why don't we just add empty maps here? because we don't
        // know the nc.count for these names
        map[SOURCES_KEY] = srcNameList;
        compoundSrcNameList.push(...srcNameList);
        compoundNcList.push(nc);
    }, this);

    if (args.allPrimary && (primarySrcNameList.length != args.ncList.length)) {
        throw new Error(`something went wrong, primary: ${primarySrcNameList.length}` +
                        `, ncList: ${args.ncList.length}`);
    }

    Debug(`--buildSrcNameList`);
    Debug(`  compoundSrcNameList: ${compoundSrcNameList}`);
    Debug(`  compoundNcList: ${compoundNcList}`);
    Debug(`  count: ${clueCount}`);
    Debug(`  primarySrcNameList: ${primarySrcNameList}`);
    Debug(`__primaryNcList: ${primaryNcList}`);
    // indentNewline() + '  compoundNcNameListPairs: ' + compoundNcNameListPairs +

    if (!_.isEmpty(resultMap.map())) {
        Debug(`resultMap:`);
        resultMap.dump();
    } else {
        Debug(`resultMap: empty`);
    }
    return {
        compoundNcNameListPairs,
        compoundSrcNameList,
        compoundNcList,
        primaryNcList,
        primarySrcNameList,
        resultMap,
        allPrimary,
        indexMap,
        count: clueCount
    };
};

//
//

let getIndexMap = (indexMap: any): any => {
    if (!_.isUndefined(indexMap)) {
        Debug(`using index map`);
        return indexMap;
    }
    Debug(`new index map`);
    return {};
};

//
//

let getSrcListIndex = (indexMap: any, nc: /*NameCount*/ any, srcList: number[]): number => {
    let slIndex;
    if (_.has(indexMap, nc)) {
        slIndex = indexMap[nc].index;
        // sanity check
        if (xp) Expect(indexMap[nc].length, 'mismatched index lengths').is.equal(srcList.length);
        Debug(`${nc.name}: using preset index ${slIndex}` +
                 `, length(${indexMap[nc].length})` +
                 `, actual length(${srcList.length})`);
    } else {
        slIndex = 0;
        indexMap[nc] = { index: 0, length: srcList.length };
        Debug(`${nc.name}: using first index ${slIndex}` +
              `, actual length(${srcList.length})`);
    }
    return slIndex;
};

//
//
let incrementIndexMap = (indexMap: any): boolean => {
    if (xp) Expect(indexMap).is.an.Object().and.not.empty();
    Debug(`++indexMap: ${indexMapToJSON(indexMap)}`);

    // TODO: this is a bit flaky. assumes the order of keys isn't changing.
    let keyList = Object.keys(indexMap);

    // start at last index
    let keyIndex = keyList.length - 1;
    let indexObj = indexMap[keyList[keyIndex]];
    indexObj.index += 1;

    // while index is maxed reset to zero, increment next-to-last index, etc.
    // using >= because it's possible both index and length are zero
    // for primary clues, which are skipped.
    while (indexObj.index >= indexObj.length) {
        Debug(`keyIndex ${keyIndex}: ${indexObj.index}` +
              ` >= ${indexObj.length}, resetting`);
        indexObj.index = 0;
        keyIndex -= 1;
        if (keyIndex < 0) {
            return false;
        }
        indexObj = indexMap[keyList[keyIndex]];
        indexObj.index += 1;
        Debug(`keyIndex ${keyIndex}: ${indexObj.index}` +
              `, length: ${indexObj.length}`);
    }
    Debug(`--indexMap: ${indexMapToJSON(indexMap)}`);
    return true;
};

//
//
let indexMapToJSON = (map: any): string => {
    let s = '';
    _.keys(map).forEach((key: any) => {
        if (s.length > 0) {
            s += ',';
        }
        s += map[key].index;
    });
    return '[' + s + ']';
};

//
//

let copyAddNcList = (ncList: NCList, name: string, count: number): NCList => {
    // for non-primary check for duplicate name:count entry
    // technically this is allowable for count > 1 if the there are
    // multiple entries of this clue name in the clueList[count].
    // (at least as many entries as there are copies of name in ncList)
    // TODO: make knownSourceMapArray store a count instead of boolean

    if (!ncList.every((nc: NameCount) => {
        if (nc.count > 1) {
            if ((name === nc.name) && (count === nc.count)) {
                return false;
            }
        }
        return true;
    })) {
        return [];
    }

    // TODO: _.clone()
    let newNcList = Array.from(ncList);
    newNcList.push(NameCount.makeNew(name, count));
    return newNcList;
}

//
//

let getDiffNcList = (origNcList: NCList, nameCountList: NCList): NCList => {
    let ncList: NCList = [];
    for (let index = origNcList.length; index < nameCountList.length; ++index) {
        ncList.push(nameCountList[index]);
    }
    return ncList;
};

//
//
let getNameSrcList = (srcMap: any): NCList => {
    return _.keys(srcMap).map(key => NameCount.makeNew(srcMap[key], key));
};

//
//
let chop = (list: any, removeValue: any): any => {
    let copy: any[] = [];
    list.forEach((value: any) => {
        if (value == removeValue) {
            removeValue = undefined;
        } else {
            copy.push(value);
        }
    });
    return copy;
};

//
//
let hasNameSrcList = (resultList: any, nameSrcList: NCList): boolean => {
    return resultList.some((result: any) => {
        return result.nameSrcList.every((nameSrc, nsIndex) => {
            return nameSrc.equals(nameSrcList[nsIndex]);
        });
    });
};

// args:
//  nameSrcList:
//  excludeSrcList: list of excluded primary sources
//
// TODO: NameCount.containsAnyCount(ncList-or-nc, count-or-countlist)
//
/*
let hasExcludedSource = function(nameSrcList: NCList, excludeSrcList: NCList) {
    return _.isUndefined(excludeSrcList) ? false :
        !_.isEmpty(_.intersection(excludeSrcList, nameSrcList.map(nc => nc.count)));
};
*/

//
//

let dumpIndexMap = function(indexMap: any): void {
    let s = '';
    _.keys(indexMap).forEach((key: any) => {
        let entry = indexMap[key];
        if (s.length > 0) {
            s += '; ';
        }
        s += 'index ' + entry.index + ', length ' + entry.length;
    });
    Debug(s);
};

// args:
//  nameList:       list of clue names, e.g. ['bob','jim']
//  sum:            # of primary clues represented by names in nameList
//  max:            max # of sources to combine (either max -or- count must be set)
//  count:          exact # of sources to combine (either count -or- max must be set)
//  validateAll:    flag; check all combinations
//  quiet:          flag; quiet Peco
//
// All the primary clues which make up the clues in /nameList/ should
// be unique and their total count should add up to /count/. Verify
// that some combination of the cluelists of all possible addends of
// /count/ makes this true.

// args:
//   name:           clueName,
//   count:          # of primary clues represented by cluename
//   nameList:
//   countList:
//   ncList:
//   validateAll:
///
let checkUniqueSources = (nameCountList: NCList, args: any): any => {
    let origNcList = nameCountList;

    // assert(nameCountList) && Array.isArray(nameCountList)

    Debug('++checkUniqueSouces' +
          `, name: ${args.name}, count: ${args.count}, nameList: ${args.nameList}` +
          `, validateAll: ${args.validateAll}, ncList: ${args.nameCountList}`);
    
    // first, check for duplicate primary clues, add all to primaryMap
    let findResult = findDuplicatePrimaryClue({ ncList: nameCountList });
    if (!evalFindDuplicateResult(findResult, '1st')) {
        Debug(`FAIL, duplicate primary, nameCountList: ${nameCountList}`);
        return uniqueResult(false); // failure
    }
    else if (findResult.allPrimary) {
        let nameSrcList = getNameSrcList(findResult.srcMap);
        let resultMap = ResultMap.makeNew();
        resultMap.addPrimaryLists(nameCountList, nameSrcList);
        let compatList = getCompatibleResults({
            origNcList:     nameCountList,
            ncList:         nameCountList,
            nameSrcList,
            pendingMap:     resultMap,
            validateAll:    args.validateAll
        });
        if (!_.isEmpty(compatList)) {
            return uniqueResult(true, compatList);
        }
        return uniqueResult(false);
    }

    let resultMap;
    let buildResult;
    let candidateResultList;
    let anyFlag = false;
    let resultList: any[] = [];
    let buildArgs: any = {
        ncList: nameCountList
    };

    for(;;) {
        if (buildResult) {
            buildArgs.indexMap = buildResult.indexMap;
        }
        nameCountList = origNcList;

        for (;;) {
            // break down all compound clues into source components
            buildArgs.ncList = nameCountList;
            buildResult = buildSrcNameList(buildArgs);
            if (xp) Expect(buildResult.count).is.belowOrEqual(ClueManager.maxClues);

            // skip recursive call to validateSources if we have all primary clues
            if (buildResult.allPrimary) {
                Debug('cUS: adding all_primary result: ' +
                      `${nameCountList} = ${buildResult.primaryNcList}`);
                nameCountList = buildResult.primaryNcList;
                candidateResultList = [{
                    ncList:    buildResult.primaryNcList,
                    resultMap: buildResult.resultMap // no need to merge?
                }];
            } else {
                // call validateSources recursively with compound clues
                let vsResult = validateSources({
                    sum:            buildResult.count,
                    nameList:       buildResult.compoundSrcNameList,
                    count:          buildResult.compoundSrcNameList.length,
                    validateAll:    true  // always validate all on recursive call
                });
                if (!vsResult.success) {
                    break; // fail, try other combos
                }
                // sanity check
                if (xp) Expect(vsResult.list).is.not.empty();
                Debug(`from validateSources(${buildResult.count$})`);
                Debug(`  compoundSrcNameList: ${buildResult.compoundSrcNameList}`);
                Debug(`  compoundNcList: ${buildResult.compoundNcList}`);
                Debug(`  list.size: ${_.size(vsResult.list)}`);
                vsResult.list.forEach((result: any) => {
                    Debug(`   ncList:      ${result.ncList}`);
                    Debug(`   nameSrcList: ${result.nameSrcList}`);
                    Debug(`   -----------`);
                });
                // we only sent compound clues to validateSources, so add the primary
                // clues that were filtered out by build(), to make a complete list.
                // also merge buildResults data into resultMap.
                candidateResultList = vsResult.list.map(result => Object({
                    ncList:    _.concat(result.ncList, buildResult.primaryNcList),
                    resultMap: _.cloneDeep(buildResult.resultMap).merge(result.resultMap, buildResult.compoundNcList)
                }));
            }

            let anyCandidate = false;
            candidateResultList.some((result: any) => {
                let findResult = findDuplicatePrimaryClue({ ncList: result.ncList });
                if (xp) Expect(findResult.allPrimary).is.true();
                if (!evalFindDuplicateResult(findResult, '2nd')) {
                    return false; // some.continue
                }
                let compatList = getCompatibleResults({
                    origNcList:     buildArgs.ncList,
                    ncList:         result.ncList,
                    nameSrcList:    getNameSrcList(findResult.srcMap),
                    pendingMap:     result.resultMap, 
                    validateAll:    args.validateAll,
                    ncNameListPairs: buildResult.compoundNcNameListPairs // TODO: remove here and in build()
                });
                if (!_.isEmpty(compatList)) {
                    anyCandidate = true;
                    compatList.forEach((result: any) => {
                        if (!hasNameSrcList(resultList, result.nameSrcList)) {
                            resultList.push(result);
                        }
                    });
                    // TODO: remove duplicates in uniqueResults()
                    //resultList = _.concat(resultList, compatList);
                    return !args.validateAll; // some.exit if !validateAll, else some.continue
                }
                return false; // some.continue;
            });
            if (!anyCandidate) {
                break; // none of those results were good, try other combos
            }
            anyFlag = true;
            if (args.validateAll) {
                break; // success , but keep searching for other combos
            }
            Debug(`--checkUniqueSources, single validate, success: ${anyFlag}`);
            return uniqueResult(anyFlag, resultList); // success , exit function
        }
        // sanity check
        if (xp) Expect(buildResult).is.ok();
        if (!incrementIndexMap(buildResult.indexMap)) {
            Debug(`--checkUniqueSources, full validate, success: ${anyFlag}`);
            return uniqueResult(anyFlag, resultList);
        }
        Debug(`++outer looping`);
    }
};

// args:
//   name     : clueName,
//   count    : count,
//   nameList : clueNameList,
//   countList: clueCountList,
//   ncList
//   validateAll:
//
// returns:
//   success  : boolean
//   list     : list
//
// TODO: ForName

let rvsWorker = (args: any): any => {
    Debug('++rvsWorker' +
          `, name: ${args.name}` +
          `, count: ${args.count}` +
          `, validateAll: ${args.validateAll}` +
          `${indentNewline()}  ncList: ${args.ncList}` +
          `, nameList: ${args.nameList}`);
    if (xp) {
        Expect(args.name).is.a.String().and.not.empty();
        Expect(args.count).is.a.Number()
            .aboveOrEqual(1)
            .belowOrEqual(ClueManager.maxClues);
        Expect(args.ncList).is.an.Array();
        Expect(args.nameList).is.an.Array();
    }
        
    let newNameCountList = copyAddNcList(args.ncList, args.name, args.count);
    if (_.isEmpty(newNameCountList)) {
        // TODO:
        // duplicate name:count entry. technically this is allowable for
        // count > 1 if the there are multiple entries of this clue name
        // in the clueList[count]. (at least as many entries as there are
        // copies of name in ncList)
        // SEE ALSO: copyAddNcList()
        Debug(`++rvsWorker, duplicate name:count, ${args.name}:{args.count}`);
        return { success: false }; // fail
    }
    Debug(`added NC name: ${args.name}, count: ${args.count}, list.length: ${newNameCountList.length}`);
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
        let result = checkUniqueSources(newNameCountList, args);
        Debug(`checkUniqueSources --- ${result.success ? 'success!' : 'failure'}`);
        if (result.success) {
            args.ncList.push(NameCount.makeNew(args.name, args.count));
            Debug(`add1, ${args.name}:${args.count}` +
                  `, newNc(${newNameCountList.length})` +
                  `, ${newNameCountList}`);
        }
        return result;
    }

    // nameList.length > 1, remove current name & count,
    // and validate remaining
    Debug(`calling rvs recursively, ncList: ${newNameCountList}`);
    let rvsResult = recursiveValidateSources({
        clueNameList:  chop(args.nameList, args.name),
        clueCountList: chop(args.countList, args.count),
        nameCountList: newNameCountList,
        validateAll:   args.validateAll
    });
    if (!rvsResult.success) {
        Debug('--rvsWorker, recursiveValidateSources failed');
        return { success: false }; // fail
    }
    // does this achieve anything? modifies args.ncList.
    // TODO: probably need to remove why that matters.
    // TODO2: use _clone() until then
    args.ncList.length = 0;
    newNameCountList.forEach((nc: NameCount) => args.ncList.push(nc));
    Debug(`--rvsWorker, add ${args.name}:${args.count}` +
          `, newNcList(${newNameCountList.length})` +
          `, ${newNameCountList}`);
    return {
        success: true,
        list:    rvsResult.list
    };
};

// args:
//  clueNameList:
//  clueCountList:
//  nameCountList:
//  validateAll:
//
// TODO: ForNameList
let recursiveValidateSources = (args: any): RvsResult => {
    if (xp) {
        Expect(args.clueNameList).is.an.Array().and.not.empty();
        Expect(args.clueCountList).is.an.Array().and.not.empty();
    }

    logLevel++;
    Debug(`++recursiveValidateSources, looking for [${args.clueNameList}]` +
          ` in [${args.clueCountList}]`);
    if (xp) Expect(args.clueNameList.length).is.equal(args.clueCountList.length);

    let ncList = args.nameCountList || [];
    let nameIndex = 0;
    let clueName = args.clueNameList[nameIndex];
    let resultList;

    // optimization: could have a map of count:boolean entries here
    // on a per-name basis (new map for each outer loop; once a
    // count is checked for a name, no need to check it again

    let someResult = args.clueCountList.some((count: number) => {
        Debug(`looking for ${clueName} in ${count}`);
        if (!_.has(ClueManager.knownClueMapArray[count], clueName)) {
            Debug(` not found, ${clueName}:${count}`);
            return false; // some.continue
        }
        Debug(' found');
        let rvsResult = rvsWorker({
            name:           clueName,
            count:          count,
            nameList:       args.clueNameList,
            countList:      args.clueCountList,
            ncList:         ncList,
            validateAll:    args.validateAll
        });
        if (!rvsResult.success) {
            return false; // some.continue;
        }
        Debug(`  rvsWorker output for: ${clueName}, ncList(${ncList.length}) ${ncList}`);
        // sanity check
        if (!args.validateAll && (ncList.length < 2)) {
            // TODO: add "allowSingleEntry" ?
            // can i check vs. clueNameList.length?
            // throw new Error('list should have at least two entries1');
        }
        resultList = rvsResult.list;
        return true; // success: some.exit
    });
    --logLevel;

    return {
        success: someResult,
        list:    someResult ? resultList : undefined
    };
};

let hash = {};
let hash_size = 0;

//
//
let validateSources = (args: any): any => {
    Debug('++validateSources' +
          `${indentNewline()}  nameList(${args.nameList.length}): ${args.nameList}` +
          `, sum(${args.sum})` +
          `, count(${args.count})` +
          `, validateAll: ${args.validateAll}`);

    ++count;
    let is_dupe = false;
    let key = `${args.namelist}:${args.sum}:${args.count}`;
    if (key in hash) {
	dupe += 1;
	is_dupe = true;
	const result = hash[key];
	//if (!result.success) return result;
    } else {
	hash_size += 1;
    }
    if (!(count % 10000)) {
	Timing(`++validateSources (${++count}): ${args.nameList}, hash(${hash_size}) ${is_dupe ? '(dupe)' : ''}`);
    }
    let found = false;
    let resultList: Result[] = [];
    Peco.makeNew({
        sum:     args.sum,
        count:   args.count,
        max:     args.max,
        quiet:   args.quiet
    }).getCombinations().some((clueCountList: number[]) => {
        let rvsResult = recursiveValidateSources({
            clueNameList:   args.nameList,
            clueCountList,
            validateAll:    args.validateAll
        });
        if (rvsResult.success) {
            Debug('validateSources: VALIDATE SUCCESS!');
            resultList.push(...rvsResult.list);
            found = true;
            if (!args.validateAll) return true; // found a match; some.exit
            // validatingg all, continue searching
            Debug('validateSources: validateAll set, continuing...');
        }
        return false; // some.continue
    }, this);
    Debug('--validateSources');

    const result = {
        success:     found,
        list:        found ? resultList : undefined
    };

    //hash[key] = result;
    return result;
};

module.exports = {
    setAllowDupeFlags,
    validateSources
};
