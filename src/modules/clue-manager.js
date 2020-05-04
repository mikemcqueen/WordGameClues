//
// clue-manager.js
//

'use strict';

// export a singleton

module.exports = new ClueManager();

const _              = require('lodash');
const ClueList       = require('../types/clue-list');
const Clues          = require('./clue-types');
const Debug          = require('debug')('clue-manager');
const Expect         = require('should/as-function');
const Log            = require('./log')('clue-manager');
const NameCount      = require('../types/name-count');
const Path           = require('path');
const Peco           = require('./peco');
const Stringify      = require('stringify-object');
const Validator      = require('./validator');

//

const DATA_DIR              =  Path.normalize(`${Path.dirname(module.filename)}/../../data/`);
const REJECTS_DIR           = 'rejects';

// constructor

function ClueManager () {
    this.clueListArray = [];         // the JSON clue files in an array
    this.maybeListArray = [];        // the JSON maybe files in an array
    this.rejectListArray = [];       // the JSON reject files in an array
    this.knownClueMapArray = [];     // map clue name to clue src
    this.knownSourceMapArray = [];   // map known source to list of clues
    this.rejectSourceMap = {};       // map reject source to true/false (currently)

    this.rvsSuccessSeconds = 0;
    this.rvsFailDuration = 0;

    this.loaded = false;
    this.maxClues = 0;

    this.logging = false;
//    this.logging = true;

    this.logLevel = 0;
}

//

ClueManager.prototype.log = function (text) {
    let pad = '';
    let index;
    if (this.logging) {
        for (let index = 0; index < this.logLevel; index += 1) { pad += ' '; }
        console.log(pad + text);
    }
}

//


ClueManager.prototype.saveClueList = function (list, count, options = {}) {
    list.save(this.getKnownFilename(count, options.dir));
}

// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//

ClueManager.prototype.loadAllClues = function (args) {
    this.dir = Clues.getDirectory(args.clues);
    if (args.ignoreErrors) {
        this.ignoreLoadErrors = true;
    }
    this.maxClues = args.clues.clueCount;

    for (let count = 1; count <= this.maxClues; ++count) {
        let knownClueList = this.loadClueList(count);
        this.clueListArray[count] = knownClueList;
        if (count === 1) {
            this.addKnownPrimaryClues(knownClueList);
        }
        else {
            this.addKnownCompoundClues(knownClueList, count, args.validateAll);
        }
    }

    for (let count = 2; count <= this.maxClues; ++count) {
        let rejectClueList;
        try {
            rejectClueList = ClueList.makeFrom({
                filename : this.getRejectFilename(count)
            });
        }
        catch (err) {
            console.log(`WARNING! reject file: ${this.getRejectFilename(count)}, ${err}`);
        }
        if (rejectClueList) {
            this.rejectListArray[count] = rejectClueList;
            this.addRejectCombos(rejectClueList, count);
        }
    }

    this.loaded = true;

    return this;
}

//

ClueManager.prototype.loadClueList = function (count, options = {}) {
    return ClueList.makeFrom({
        filename : this.getKnownFilename(count, options.dir)
    });
}

//

ClueManager.prototype.addKnownPrimaryClues = function (clueList) {
    const count = 1;
    let clueMap = this.knownClueMapArray[count] = {};
    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
        if (!_.has(clueMap, clue.name)) {
            clueMap[clue.name] = [];
        }
        clueMap[clue.name].push(clue.src);
    });
    return this;
}

//

ClueManager.prototype.getKnownFilename = function (count, dir = undefined) {
    return Path.format({
        dir:  _.isUndefined(dir) ? this.dir : `${DATA_DIR}${dir}`,
        base: `clues${count}.json`
    });
}

//

ClueManager.prototype.getRejectFilename = function (count) {
    return Path.format({
        dir:  `${DATA_DIR}${REJECTS_DIR}`,
        base: `rejects${count}.json`
    });
}

//

ClueManager.prototype.addKnownCompoundClues = function (clueList, clueCount, validateAll) {
    // so this is currently only callable once per clueCount.
    Expect(this.knownClueMapArray[clueCount]).is.undefined();
    this.knownClueMapArray[clueCount] = {};
    if (clueCount > 1) {
        Expect(this.knownSourceMapArray[clueCount]).is.undefined();
        this.knownSourceMapArray[clueCount] = {};
    }
    let srcMap = this.knownSourceMapArray[clueCount];
    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
        let srcNameList = clue.src.split(',').sort();
        let srcKey = srcNameList.toString();
        if (clueCount > 1) {
            // new sources need to be validated
            if (!_.has(srcMap, srcKey)) {
                Debug(`## validating Known Combo: ${srcKey}:${clueCount}`);
                let vsResult = Validator.validateSources({
                    sum:         clueCount,
                    nameList:    srcNameList,
                    count:       srcNameList.length,
                    validateAll: validateAll
                });
                if (!this.ignoreLoadErrors) {
		    if (!vsResult.success) {
			console.log(`failed srcNameList: ${srcNameList}`);
		    }
                    Expect(vsResult.success);
                }
                srcMap[srcKey] = [];
            }
            srcMap[srcKey].push(clue);
        }
        this.addKnownClue(clueCount, clue.name, srcKey);
    }, this);
    return this;
}

//

ClueManager.prototype.addKnownClue = function (count, name, source, nothrow) {
    Expect(count).is.a.Number();
    Expect(name).is.a.String();
    Expect(source).is.a.String();
    let clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, name)) {
        clueMap[name] = [ source ];
    } else if (!clueMap[name].includes(source)) {
        if (this.logging) {
            this.log(`clueMap[${name}] = ${clueMap[name]}`);
            this.log('addKnownClue(' + count + ') ' +
                     name + ' : ' + source);
        }
        clueMap[name].push(source);
    } else {
        if (nothrow) return false;
        throw new Error('duplicate clue name/source' + 
                        '(' + count + ') ' +
                        name + ' : ' + source);
    }
    return true;
}

//

ClueManager.prototype.removeKnownClue = function (count, name, source, nothrow) {
    Expect(count).is.a.Number();
    Expect(name).is.a.String();
    Expect(source).is.a.String();
    let clueMap = this.knownClueMapArray[count];
    if (!_.has(clueMap, name) || !clueMap[name].includes(source)) {
        if (nothrow) return false;
        throw new Error(`removeKnownClue, missing clue: ${name}:${source} at count: ${count}`);
    } else {
        Debug(`before clueMap[${name}]: len(${clueMap[name].length}), sources:${clueMap[name]}`);
        Debug(`removing clue: [${name}] : ${source} from count: ${count}`);
        _.pull(clueMap[name], source);
        Debug(`after clueMap[${name}]: len(${clueMap[name].length}), sources: ${clueMap[name]}`);
    }
    return true;
}

//

ClueManager.prototype.saveClues = function (counts) {
    if (_.isNumber(counts)) {
        counts = [ counts ];
    }
    Debug(`saving clueLists ${counts}`);
    Expect(counts).is.an.Array();
    for (const count of counts) {
        this.saveClueList(this.clueListArray[count], count);
        Debug(`saved clueList ${count}, length: ${this.clueListArray[count].length}`);
    }
}

//

ClueManager.prototype.addClue = function (count, clue, save = false, nothrow = false) {
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (this.addKnownClue(count, clue.name, clue.src, nothrow)) {
        this.clueListArray[count].push(clue);
        if (save) {
            this.saveClues(count);
        }
        return true;
    }
    return false;
}

//

ClueManager.prototype.removeClue = function (count, clue, save = false, nothrow = false) {
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (this.removeKnownClue(count, clue.name, clue.src, nothrow)) {
        _.remove(this.clueListArray[count], elem => {
            return (elem.name === clue.name) && (elem.src === clue.src);
        });
        if (save) {
            this.saveClues(count);
        }
        return true;
    }
    return false;
}

//

ClueManager.prototype.addMaybe = function (name, srcNameList, note, save = false) {
    return false;
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array();
    let count = _.size(srcNameList);
    Expect(count).is.above(1); // at.least(2);
    let clue = {
        name: name,
        src: _.toString(srcNameList)
    };
    if (!_.isUndefined(note)) {
        clue.note = note;
    }
    this.maybeListArray[count].push(clue);
    if (save) {
        this.maybeListArray[count].save(this.getMaybeFilename(count));
    }
    return true;
}

//

ClueManager.prototype.addRejectCombos = function (clueList, clueCount) {
    clueList.forEach(clue => {
        let srcNameList = clue.src.split(',');
        if (_.size(srcNameList) !== clueCount) {
            this.log(`WARNING! reject word count mismatch` +
                     `, expected {clueCount}, actual ${_.size(srcNameList)}, ${srcNameList}`);
        }
        this.addRejectSource(srcNameList);
    });
    return this;
}

//

ClueManager.prototype.saveRejects = function (counts) {
    if (_.isNumber(counts)) {
        counts = [ counts ];
    }
    Expect(counts).is.an.Array();
    counts.forEach(count => {
        this.rejectListArray[count].save(this.getRejectFilename(count));
    });
}

//

ClueManager.prototype.addReject = function (srcNameList, save = false) {
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array();
    let count = _.size(srcNameList);
    Expect(count).is.above(1); // at.least(2);
    if (this.addRejectSource(srcNameList)) {
        this.rejectListArray[count].push({
            src: _.toString(srcNameList)
        });
        if (save) {
            this.rejectListArray[count].save(this.getRejectFilename(count));
        }
        return true;
    }
    return false;
}

//

ClueManager.prototype.addRejectSource = function (srcNameList) {
    if (_.isString(srcNameList)) {
        srcNameList = srcNameList.split(',');
    }
    Expect(srcNameList).is.an.Array().and.not.empty();
    srcNameList.sort();
    if (this.logging) {
        this.log('addRejectSource: ' + srcNameList);
    }

    if (this.isKnownSource(srcNameList.toString())) {
        console.log('WARNING! not rejecting known source, ' + srcNameList);
        return false;
    }
    if (this.isRejectSource(srcNameList)) {
        console.log('WARNING! duplicate reject source, ' + srcNameList);
        // i had this return false commented out for some reason,
        // but it should be here.
        return false;
    }
    this.rejectSourceMap[srcNameList.toString()] = true;
    return true;
}

// source is string containing sorted, comma-separated clues
//
ClueManager.prototype.isKnownSource = function (source, count = 0) {
    Expect(source).is.a.String();
    Expect(count).is.a.Number();

    // check for supplied count
    if (count > 0) {
        return _.has(this.knownSourceMapArray[count], source);
    }
    // check for all counts
    return this.knownSourceMapArray.some(srcMap => _.has(srcMap, source));
}

// source: csv string or array of strings

ClueManager.prototype.isRejectSource = function (source) {
    if (!_.isString(source) && !_.isArray(source)) {
        throw new Error('bad source: ' + source);
    }
    return _.has(this.rejectSourceMap, source.toString());
}

//

ClueManager.prototype.getCountListForName = function (name) {
    let countList = [];
    for (const [index, clueMap] of this.knownClueMapArray.entries()) {
        if (_.has(clueMap, name)) {
            countList.push(index);
        }
    };
    return countList;
}

//

ClueManager.prototype.getSrcListForNc = function (nc) {
    let clueMap = this.knownClueMapArray[nc.count];
    Expect(_.has(clueMap, nc.name)).is.true();
    return clueMap[nc.name];
}

//

ClueManager.prototype.getSrcListMapForName = function (name) {
    let srcListMap = {};
    for (let index = 1; index < this.maxClues; ++index) {
        let srcList = this.knownClueMapArray[index][name];
        if (srcList) {
            srcListMap[index] = srcList;
        }
    }
    return srcListMap;
}

//

ClueManager.prototype.makeSrcNameListArray = function (nc) {
    let srcNameListArray = [];
    this.getSrcListForNc(nc).forEach(src => {
        srcNameListArray.push(src.split(','));
    });
    return srcNameListArray;
}

// args:
//  sum:     args.sum,
//  max:     args.max,
//  require: args.require
//
// A "clueSourceList" is a list (array) where each element is a
// cluelist, such as [clues1,clues1,clues2].
//
// Given a sum, such as 3, generate an array of lists of addends that
// that add up to that sum, such as [ [1, 2], [2, 1] ], and return an
// array of lists of clueLists of the corresponding clue counts, such
// as [ [clues1, clues2], [clues2, clues1] ].

ClueManager.prototype.getClueSourceListArray = function (args) {
    Log.info(`++clueSrcListArray` +
             `, sum: ${args.sum}, max: ${args.max}, require: ${args.require}`);

    let clueCountListArray = Peco.makeNew({
        sum:     args.sum,
        max:     args.max,
        require: args.require
    }).getCombinations(); 

    Debug(`clueCountListArray(${clueCountListArray.length}): ${Stringify(clueCountListArray)}`);

    let clueSourceListArray = [];
    // TODO: .filter()
    clueCountListArray.forEach(clueCountList => {
        let clueSourceList = [];
        if (clueCountList.every(count => {
            // empty lists not allowed
            if (_.isEmpty(this.clueListArray[count])) {
                return false;
            }
            clueSourceList.push({ 
                list:  this.clueListArray[count],
                count: count
            });
            return true;
        }, this)) {
            clueSourceListArray.push(clueSourceList);
        }
    }, this);
    if (!clueSourceListArray.length) {
        // this happens, for example, with -c 3 -x 3 -q1,2,3
        // we require sources 1,2,3, and search for combos with
        // -max 1,2,3 of which both -max 1,2 will hit here because
        // we can't match the 3 required sources
        //console.log('WARNING! getClueSrcListArray empty!');
    }
    return clueSourceListArray;
}

//

ClueManager.prototype.filterAddends = function (addends, sizes) {
    let filtered = [];
    addends.forEach(list => {
        if (sizes.every(size => {
            return list.indexOf(Number(size)) > -1;
        })) {
            filtered.push(list);
        }
    });
    return filtered;
}

//

ClueManager.prototype.filter = function (srcCsvList, clueCount, map = {}) {
    let known = 0;
    let reject = 0;
    let duplicate = 0;
    srcCsvList.forEach(srcCsv => {
        if (this.isRejectSource(srcCsv)) {
            if (this.logging) this.log(`isRejectSource(${clueCount}) ${srcCsv}`);
            ++reject;
        } else {
            if (this.isKnownSource(srcCsv, clueCount)) {
		if (this.logging) this.log(`isKnownSource(${clueCount}) ${srcCsv}`);
		++known;
	    }
            if (_.has(map, srcCsv)) {
                if (this.logging) this.log(`duplicate: ${srcCsv}`);
                ++duplicate;
            }
            map[srcCsv] = true;
        }
    });
    return {
        map:       map,
        known:     known,
        reject:    reject,
        duplicate: duplicate
    };
}

// this actually returns knownClueNames
// and I don't think it's returning unique values (i.e. there are duplicates);

ClueManager.prototype.old_getKnownClues = function (nameList) {
    if (_.isString(nameList)) {
        nameList = nameList.split(',');
    }
    Expect(nameList).is.an.Array();
    nameList = nameList.sort().toString();
    let resultList = [];
    this.knownSourceMapArray.forEach(srcMap => {
        if (_.has(srcMap, nameList)) {
            // srcMap[nameList] is a clueList
            resultList.push(...srcMap[nameList].map(clue => clue.name));
        }
    });
    return resultList;
}

//

ClueManager.prototype.getKnownClues = function (nameList) {
    if (_.isString(nameList)) {
        nameList = nameList.split(',');
    }
    Expect(nameList).is.an.Array();
    nameList = nameList.sort().toString();
    let nameClueMap = {};
    this.knownSourceMapArray.forEach(srcMap => {
        if (_.has(srcMap, nameList)) {
            for (const clue of srcMap[nameList]) {
                if (!_.has(nameClueMap, clue.name)) {
                    nameClueMap[clue.name] = [];
                }
                nameClueMap[clue.name].push(clue);
            }
        }
    });
    return nameClueMap;
}

//

ClueManager.prototype.getKnownClueNames = function (nameList) {
    return _.keys(this.getKnownClues(nameList));
}

//

ClueManager.prototype.getClueCountListArray = function (nameList) {
    Expect(nameList).is.not.empty();
    // each count list contains the clueMapArray indexes in which
    // each name appears
    let countListArray = Array(_.size(nameList)).fill().map(_ => []);
    for (let count = 1; count <= this.maxClues; ++count) {
        let map = this.knownClueMapArray[count];
        Expect(map).is.ok(); // I know this will fail when I move to synth clues
        nameList.forEach((name, index) => {
            if (_.has(map, name)) {
                countListArray[index].push(count);
            }
        });
    }
    return countListArray;
}

//

ClueManager.prototype.getValidCounts = function (nameList, countListArray) {
    if ((nameList.length === 1) || this.isRejectSource(nameList)) return [];

    let addCountSet = new Set();
    let known = false;
    let reject = false;
    Peco.makeNew({
        listArray: countListArray,
        max:       this.maxClues
    }).getCombinations().forEach(clueCountList => {
        Debug(`nameList: ${nameList}, clueCountList: ${clueCountList}`);
        let sum = clueCountList.reduce((a, b) => a + b);
        // why was I passing validateAll: true here, shouldn't a single
        // validation suffice?
        if (Validator.validateSources({
            sum:         sum,
            nameList:    nameList,
            count:       nameList.length,
            validateAll: false
        }).success) {
            addCountSet.add(sum);
        }
    });
    return Array.from(addCountSet);
}

//

ClueManager.prototype.getCountList = function (nameOrList) {
    return _.isString(nameOrList)
        ? this.getCountListForName(nameOrList)
        : this.getValidCounts(nameOrList, this.getClueCountListArray(nameOrList));
}

//

ClueManager.prototype.getPrimarySources = function () {
    let primarySources = [];
    let hash = {};
    for (const clue of this.clueListArray[1]) {
        if (_.has(hash, clue.src)) continue;
        primarySources.push(clue.src);
        hash[clue.src] = true;
    }
    return primarySources;
}

//

ClueManager.prototype.getInversePrimarySources = function (sources) {
    Expect(sources).is.an.Array();
    let inverseSources = [];
    for (const src of this.getPrimarySources()) {
        if (_.includes(sources, src)) continue;
        inverseSources.push(src);
    }
    return inverseSources;
}

//

ClueManager.prototype.addClueForCounts = function (countSet, name, src, options) {
    Expect(countSet).is.instanceof(Set);
    Expect(name).is.a.String();
    Expect(src).is.a.String();
    return Array.from(countSet).reduce((added, count) => {
        if (this.addClue(count, {
            name: name,
            src:  src
        }, options.save, true)) { // save, nothrow
            console.log(`${count}: added ${name}`);
	    added += 1;
        } else {
            console.log(`${count}: ${name} already present`);
	}
	return added;
    }, 0);
}

//

ClueManager.prototype.removeClueForCounts = function (countSet, name, src, options = {}) {
    Expect(countSet).is.instanceof(Set);
    Expect(name).is.a.String();
    Expect(src).is.a.String();
    let removed = 0;
    for (let count of countSet.keys()) {
        if (this.removeClue(count, {
            name: name,
            src:  src
        }, options.save, options.nothrow)) {
            Debug(`removed ${name}:${count}`);
            removed += 1;
        } else {
            // not sure this should ever happen. removeClue throws atm.
            Debug(`${name}:${count} not present`);
        }
    }
    return removed;
}

// each count list contains the clueMapArray indexes in which
// each name appears

ClueManager.prototype.getKnownClueIndexLists = function (nameList) {
    let countListArray = Array(_.size(nameList)).fill().map(_ => []);
    //Debug(countListArray);
    for (let count = 1; count <= this.maxClues; ++count) {
        const map = this.knownClueMapArray[count];
        if (!_.isUndefined(map)) {
            nameList.forEach((name, index) => {
                if (_.has(map, name)) {
                    countListArray[index].push(count);
                }
            });
        }
        else {
            console.log('missing known cluemap #' + count);
        }
    }
    // verify that all names were found
    nameList.forEach((name, index) => {
        Expect(countListArray[index]).is.ok();
    });
    return countListArray;
}

//
// args:
//   add=Name
//   remove=Name
//   reject
//   isKnown
//   isReject
//

ClueManager.prototype.addRemoveOrReject = function (args, nameList, countSet, options = {}) {
    let count = 0;
    if (args.add) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --add due to single source');
        } else if (args.isReject) {
            console.log('WARNING! cannot add known clue: already rejected, ' + nameList);
        } else {
            count = this.addClueForCounts(countSet, args.add, nameList.toString(), options);
        }
    } else if (args.remove) {
        Debug(`remove [${args.remove}] as ${nameList} from ${[...countSet.values()]}`);
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --remove due to single source');
        } else {
            let removeOptions = { save: options.save, nothrow: true };
            count = this.removeClueForCounts(countSet, args.remove, nameList.toString(), removeOptions);
        }
    } else if (args.reject) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --reject due to single source');
        } else if (args.isKnown) {
            console.log('WARNING! cannot add reject clue: already known, ' + nameList);
        } else if (this.addReject(nameList.toString(), true)) {
            console.log('updated');
        }
        else {
            console.log('update failed');
        }
    }
    return count;
}

// Probably not the most unique function name possible.

ClueManager.prototype.getCountListArrays = function (nameCsv, options) {
    const validateAll = options.any ? false : true;
    const nameList = nameCsv.split(',').sort();
    Debug(`getValidCountLists for ${nameList}`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    let countListArray = this.getKnownClueIndexLists(nameList);
    Debug(countListArray);
    let resultList = Peco.makeNew({
        listArray: countListArray,
        max:       this.maxClues
    }).getCombinations();
    if (_.isEmpty(resultList)) {
        Debug('No matches');
        return null;
    }

    let addRemoveSet;
    if (options.add || options.remove) {
        addRemoveSet = new Set();
    }
    let valid = [];
    let known = [];
    let rejects = [];
    let clues = [];
    let invalid = [];

    for (const clueCountList of resultList) {
        const sum = clueCountList.reduce((a, b) => a + b);
        const result = Validator.validateSources({
            sum:         sum,
            nameList:    nameList,
            count:       nameList.length,
            validateAll
        });
        
        if (!result.success) {
            invalid.push(clueCountList);
        } else if (this.isRejectSource(nameList)) {
            rejects.push(clueCountList);
        } else if (nameList.length === 1) {
            let name = nameList[0];
            let nameSrcList = this.clueListArray[sum]
                    .filter(clue => clue.name === name)
                    .map(clue => clue.src);
            if (nameSrcList.length > 0) {
                //let clueNameList = this.clueListArray[sum].map(clue => clue.name);
                //if (clueNameList.includes(name)) {
                //
                
                /*
                 this.clueListArray[sum].forEach(clue => {
                 if (clue.name === name) {
                 clueSrcList.push(`"${clue.src}"`);
                 }
                 });
                 */
                clues.push({ countList: clueCountList, nameList: nameSrcList });
            }
        } else {
            let clueList = this.knownSourceMapArray[sum][nameList];
            if (clueList) {
                known.push({ countList: clueCountList, nameList: clueList.map(clue => clue.name) });
            } else {
                valid.push(clueCountList);
            }
            if (options.add || options.remove) {
                addRemoveSet.add(sum);
            }
        }
    }
    return { valid, known, rejects, invalid, clues, addRemoveSet };
}
