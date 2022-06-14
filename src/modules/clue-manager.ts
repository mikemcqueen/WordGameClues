//
// clue-manager.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Log            = require('../../modules/log')('clue-manager');
const Peco           = require('../../modules/peco');
const Clues          = require('../../modules/clue-types');

const Assert = require('assert');
const Validator      = require('./validator');
const Debug          = require('debug')('clue-manager');
const Duration       = require('duration');
const Expect         = require('should/as-function');
const Path           = require('path');
const PrettyMs       = require('pretty-ms');

const Stringify2 = require('stringify-object');
const stringify = require('javascript-stringify').stringify;
//let Stringify = stringify;

import * as NameCount from '../types/name-count';
import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import type { ValidateResult, ValidateSourcesResult } from './validator';

//
//

const DATA_DIR     =  Path.normalize(`${Path.dirname(module.filename)}/../../../data/`);
const REJECTS_DIR  = 'rejects';

//
//

type CountList = number[];

// Map a clue name to a PropertyCounts Map. This is necessary because all
// clue names with the same count and source (such as synonyms) share the
// same "source data" as an optimization. What differs between them is
// whether the parent clue definition itself contains any counted properties,
// and those are stored at the root of the source data in this map.
type ClueNamePropertyCountsMapMap = Record<string, Clue.PropertyCounts.Map>;

export type SourceData = {
    clues: ClueList.Any;
    results: any[];
//    clueNamePropertyCountsMapMap: ClueNamePropertyCountsMapMap;
};

type NcResultData = {
    list: ValidateResult[];
};

type ClueMap = Record<string, string[]>;
type SourceMap = Record<string, SourceData>;
type NcResultMap = Record<string, NcResultData>;

type InternalState = {
    // TODO: typify these
    clueListArray: ClueList.Any[];    // the JSON "known" clue files in an array
    maybeListArray: any[];            // the JSON maybe files in an array
    rejectListArray: any[];           // the JSON reject files in an array
    knownClueMapArray: ClueMap[];     // map clue name to clue src
    knownSourceMapArray: SourceMap[]; // map sourceCsv to SourceData
    ncResultMapList: NcResultMap[];   // map known NCs to result list
    rejectSourceMap: any;             // map reject source to true/false (currently)
    
    dir: string;

    ignoreLoadErrors: boolean;
    loaded: boolean;
    logging: boolean;
    logLevel: number;
    
    maxClues: number;
    numPrimarySources: number;
}

//
//

let initialState = (): InternalState => {
    return {
        clueListArray: [],
        maybeListArray: [],
        rejectListArray: [],
        knownClueMapArray: [],
        knownSourceMapArray: [],
        ncResultMapList: [],
        rejectSourceMap: {},

        dir: '',
        
        ignoreLoadErrors: false,
        loaded: false,
        logging: false,
        logLevel: 0,
        
        maxClues: 0,
        numPrimarySources: 0
    };
}

let State: InternalState = initialState();

export let isLoaded = (): boolean => { return State.loaded; }
export function getMaxClues (): number { return State.maxClues; }
export function getNumPrimarySources(): number { return State.numPrimarySources; }
export function setLogging (onOff: boolean): void { State.logging = onOff; }

export function getClueList (index: number): ClueList.Any {// Compound {
    return State.clueListArray[index];
}

export function getKnownClueMap (count: number): ClueMap {
    return State.knownClueMapArray[count];
}

export function getKnownSourceMap (count: number): SourceMap {
    return State.knownSourceMapArray[count];
}

export function getNcResultMap (count: number): NcResultMap {
    return State.ncResultMapList[count];
}

//

function Stringify (val: any): string {
    return stringify(val, (value: any, indent: number, stringify: any) => {
    if (typeof value == 'function') return "function";
    return stringify(value);
    }, " ");
}

//

function showStrList (strList: string[]): string {
    let result = "";
    let first = true;
    for (let str of strList) {
        if (!first) result += ' - ';
    result += str;
        first = false;
    }
    return _.isEmpty(result) ? "[]" : result;
}

//

let log = function (text: string): void {
    let pad = '';
    let index;
    if (State.logging) {
        for (let index = 0; index < State.logLevel; index += 1) { pad += ' '; }
        console.log(pad + text);
    }
};

//

interface SaveClueListOptions {
    dir : string;
}

export let saveClueList = function (list: ClueList.Compound, count: number, options?: SaveClueListOptions): void {
    ClueList.save(list, getKnownFilename(count, options?.dir));
};

const initCluePropertyCounts = (clueList: ClueList.Primary, ignoreList: ClueList.Primary): void => {
    for (const clue of clueList) {
        Debug(`iCPC: ${Stringify(clue)}`);
        clue.propertyCounts = Clue.PropertyCounts.createMapFromClue(clue);
        const sources = clue.source?.split(',') || [];
        sources.forEach(source => {
            Debug(`iCPC: source: ${source}`);
            // the (intentional) source/source here is from hack in autoSource
            Clue.PropertyCounts.addAll(clue, _.find(ignoreList, { name: source, src: source })!);
        });
    }
};

const autoSource = (clueList: ClueList.Primary): [ClueList.Primary, number] => {
    let source = 0;
    let clueNumber = 0;
    let ignoredClues: ClueList.Primary = [];;
    let actualClues: ClueList.Primary = []
    let firstClueWithSrc: Clue.Primary | undefined = undefined;

    for (let clue of clueList) {
        // clue.num check must happen before clue.ignore check
        if (clue.num) clueNumber = Number(clue.num);
        const isSameSrc: boolean = clue.src === "same";
        if (isSameSrc) {
            // propagate the .source field from the first clue with a specific src
            // to all subsequent clues with the "same" src.
            if (clue.source) throw new Error(`${clue.name}.src="same" has source="${clue.source}"`);
            clue.source = firstClueWithSrc!.source;
        }
        if (clue.ignore) {
            if (clue.name) {
                // setting clue.src = clue.name is a hack for property count propagation from
                // dependent to parent ignored clues
                clue.src = clue.name; 
                ignoredClues.push(clue);
            }
            continue;
        }
        if (!isSameSrc) {
            source += 1;
            // save the first clue with a specific src, to enable .source field propagation
            // to subsequent clues with "same" src, above.
            firstClueWithSrc = clue;
        }
        clue.src = `${source}`;
        clue.num = clueNumber;
        actualClues.push(clue);
    }
    let loggy = false;
    initCluePropertyCounts(ignoredClues, ignoredClues);
    if (loggy) {
        console.log('Ignored:');
        ClueList.display(ignoredClues, { synonym: true });
    }
    initCluePropertyCounts(actualClues, ignoredClues);
    if (loggy) {
        console.log('Actual:');
        ClueList.display(actualClues, { synonym: true });
    }
    
    Debug(`autoSource: ${source} primary clues, ` +
        `${actualClues.reduce((srcList: string[], clue: Clue.Primary) => { srcList.push(clue.src||"undefined"); return srcList; }, [])}`);
    return [actualClues, source];
};

// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//

export let loadAllClues = function (args: any): void {
    State.dir = Clues.getDirectory(args.clues);
    if (args.ignoreErrors) {
        State.ignoreLoadErrors = true;
    }
    State.maxClues = args.max; // args.clues.clueCount;

    let primaryClueList: ClueList.Primary = loadClueList(1) as ClueList.Primary;
    if (primaryClueList[0].src === 'auto') {
    let numPrimarySources;
    [primaryClueList, numPrimarySources] = autoSource(primaryClueList);
    State.numPrimarySources = numPrimarySources;
    State.clueListArray[1] = primaryClueList;
        addKnownPrimaryClues(primaryClueList);
    } else {
        throw new Error('numPrimarySources not initialized without src="auto"');
    }

    for (let count = 2; count <= State.numPrimarySources; ++count) {
        let compoundClueList: ClueList.Compound = loadClueList(count);
    State.clueListArray[count] = compoundClueList;
        addKnownCompoundClues(compoundClueList, count, args.validateAll, args.fast);
    }

    /*
    for (let count = 2; count <= State.maxClues; ++count) {
        let rejectClueList;
        try {
            rejectClueList = ClueList.makeFrom({
                filename : State.getRejectFilename(count)
            });
        }
        catch (err) {
            console.log(`WARNING! reject file: ${State.getRejectFilename(count)}, ${err}`);
        }
        if (rejectClueList) {
            State.rejectListArray[count] = rejectClueList;
            addRejectCombos(rejectClueList, count);
        }
    }
    */
    State.loaded = true;
//    return this;
};

//

interface LoadClueListOptions {
    dir : string;
}

export let loadClueList = function (count: number, options?: LoadClueListOptions): ClueList.Any {
    const filename = getKnownFilename(count, options?.dir);
    return ClueList.makeFrom({
        filename,
        primary: count === 1
    });
};

//

let addPrimaryClueToMap = function (clue: Clue.Primary): void {
    const clueMap = getKnownClueMap(1);
    if (!_.has(clueMap, clue.name)) {
        clueMap[clue.name] = [];
    }
    clueMap[clue.name].push(clue.src);
};

//

let addKnownPrimaryClues = function (clueList: ClueList.Primary): void {
    let clueMap = State.knownClueMapArray[1] = {};
    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
        addPrimaryClueToMap(clue);
    });
};

//

let getKnownFilename = function (count: number, dir?: string): string {
    return Path.format({
        dir: !dir ? State.dir : `${DATA_DIR}${dir}`,
        base: `clues${count}.json`
    });
};

//

let getRejectFilename = function (count: number): string {
    return Path.format({
        dir:  `${DATA_DIR}${REJECTS_DIR}`,
        base: `rejects${count}.json`
    });
};

//
// key types:
// A.  non-array vlue
// B.  array value

//{
// A.
//  'jack:3': {
// A:
//    'card:2': {
// B:
//  'bird:1,red:1': [
//    'bird:2,red:8'
//  ]
//    },
// A:                       
//    'face:1': {
// B:
//  'face:1': [
//    'face:10'
//  ]
//    }
//  }
//}
//
//{
// B:
//  'face:1': [
//    'face:10'
//  ]
//}

let getResultNodePrimaryNameSrcList = function (node: any) : NameCount.List {
    let primaryNameSrcList: NameCount.List = [];
    if (_.isArray(node)) {
        Assert(node.length === 1);
        if (_.isString(node[0])) {
            primaryNameSrcList = [...node[0].split(',').map(nameSrcStr => NameCount.makeNew(nameSrcStr))];
        } else {
            primaryNameSrcList = [node[0]];
        }
    } else {
        Assert(_.isObject(node));
        for (const key of Object.keys(node)) {
            primaryNameSrcList.push(...getResultNodePrimaryNameSrcList(node[key]));
        }
    }
    return primaryNameSrcList;
}

//

let computePropertyCountsForPrimaryNameSrcList = function (primaryNameSrcList: NameCount.List,
                                                           propertyName: Clue.PropertyName.Any): Clue.PropertyCounts.Type {
    const clueList = getClueList(1);
    const propertyCounts = Clue.PropertyCounts.empty();
    for (const nc of primaryNameSrcList) {
        const clue: Clue.Primary = _.find(clueList, { name: nc.name, src: _.toString(nc.count) }) as Clue.Primary;
        Assert(clue, `bad clue: ${NameCount.toString(nc)}`);
        Clue.PropertyCounts.add(propertyCounts, clue!.propertyCounts![propertyName]);
    }
    return propertyCounts;
}

//

let getPropertyCountsForResult = function(nc: NameCount.Type, source: string,
                                          primaryNameSrcList: NameCount.List,
                                          propertyName: Clue.PropertyName.Any): Clue.PropertyCounts.Type {
    const srcData = getKnownSourceMap(nc.count)[source];
    Assert(srcData, `no SourceData for ${NameCount.makeCanonicalName(nc.name, nc.count)} - ${source}`);
    // TODO: assert: srcData.clues.find({ name: nc.name, src: source });
    const nameSrcCsv = _.sortBy(primaryNameSrcList, NameCount.count).toString();
    let propertyCounts;
    for (const result of srcData.results) {
        // TODO: nameSrcCsv is also an optional member here, I'm just not sure if it's initialized/used
        // TODO TODO TODO: sort this somewhere else.
        if (_.sortBy(result.nameSrcList, NameCount.count).toString() === nameSrcCsv) {
            propertyCounts = result.propertyCounts![propertyName];
            break;
        }
    }
    // TODO: toString(nc)
    Assert(propertyCounts, `no result for ${NameCount.toString(nc)} - ${nameSrcCsv}`);
    return propertyCounts;
}

//
//
let getResultNodePropertyCounts = function (nc: NameCount.Type, node: Object, propertyName: Clue.PropertyName.Any): Clue.PropertyCounts.Type {
    let primaryNameSrcList = getResultNodePrimaryNameSrcList(node);
    let propertyCounts: Clue.PropertyCounts.Type;
    const keys = Object.keys(node);
    if (_.isArray(node) || keys.length === 1) {
        propertyCounts = computePropertyCountsForPrimaryNameSrcList(primaryNameSrcList, propertyName);
    } else {
        const source = keys.map(key => NameCount.makeNew(key).name).sort().toString();
        propertyCounts = getPropertyCountsForResult(nc, source, primaryNameSrcList, propertyName);
    }
    return propertyCounts;
}

//
//
let computeResultPropertyCounts = function (nc: NameCount.Type, node: any,
                                            propertyName: Clue.PropertyName.Any): Clue.PropertyCounts.Type {
    let counts: Clue.PropertyCounts.Type = Clue.PropertyCounts.empty();
    for (const key of Object.keys(node)) {
        Clue.PropertyCounts.add(counts, getResultNodePropertyCounts(NameCount.makeNew(key), node[key], propertyName));
    }
    return counts;
}

//
// TODO: ForNcAndEachValidateResult
let initPropertyCountsInAllResults = (nc: NameCount.Type, results: ValidateResult[]): void => {
    for (let result of results) {
        // TODO: for name in Clue.CountedProperty.Names
        Assert(!result.propertyCounts, `already initialized PropertyCounts`);
        let propertyCounts: any = {};
        // TODO: loop over PropertyName.Enum (or PropertyName.forEach())
        propertyCounts[Clue.PropertyName.Synonym] =
            computeResultPropertyCounts(nc, result.resultMap.internal_map, Clue.PropertyName.Synonym);
        propertyCounts[Clue.PropertyName.Homonym] =
            computeResultPropertyCounts(nc, result.resultMap.internal_map, Clue.PropertyName.Homonym);
        result.propertyCounts = propertyCounts as Clue.PropertyCounts.Map;
        /*
        if (NameCount.toString(nc) === 'town:2') {
            console.error(`${NameCount.toString(nc)} init props: ${Stringify2(result.propertyCounts[Clue.PropertyName.Synonym])}`);
        }
        */
    }
};

//
// Should rename this, as it doesn't actually add a clue.
// validateCompoundClueAndUpdateState()
// 

let addCompoundClue = function (clue: Clue.Compound, count: number, validateAll = true, fast = false): ValidateSourcesResult {
    let nameList = clue.src.split(',').sort();
    let srcMap = State.knownSourceMapArray[count];
    let srcKey = nameList.toString();
    // new sources need to be validated
    let vsResult : ValidateSourcesResult = { success: true };
    if (!_.has(srcMap, srcKey)) {
        Debug(`## validating Known compound clue: ${srcKey}:${count}`);
        vsResult = Validator.validateSources({
            sum: count,
            nameList,
            count: nameList.length,
            fast,
            validateAll
        });
        // this is where the magic happens
        if (vsResult.success && validateAll) {
            initPropertyCountsInAllResults({ name: clue.name, count }, vsResult.list!);
            srcMap[srcKey] = {
                clues: [],
                results: vsResult.list!
                //,cluePropertyCountsMap: {}
            };
        }
    } else if (validateAll) {
        vsResult.list = srcMap[srcKey].results;
    }
    
    if (vsResult.success && validateAll) {
        let ncResultMap = State.ncResultMapList[count];
        if (!ncResultMap) {
            ncResultMap = State.ncResultMapList[count] = {};
        }
        // TODO: makeCanonicalName
        let ncStr = NameCount.makeNew(clue.name, count).toString();
        if (!ncResultMap[ncStr]) {
            //console.log(`adding ${ncStr} to ncResultMap`);
            ncResultMap[ncStr] = {
                list: [] // vsResult.list
            };
        }
        ncResultMap[ncStr].list.push(...vsResult.list!);
        //srcMap[srcKey].clueNamePropertyCountsMapMap[clue.name] = 
        //  Clue.PropertyCounts.createMapFromClue(clue);
        (srcMap[srcKey].clues as ClueList.Compound).push(clue);
    }
    // NOTE: added above, commented below
    // TODO: I don't understand why I'm doing this in failure case.
    // should probably be inside above if block. maybe need to split
    // out validateAll as well, i'm not sure what the "not validateAll"
    // use case is anymore. (or what .clues is use for, for that matter).
    //(srcMap[srcKey].clues as ClueList.Compound).push(clue);
    return vsResult;
};

//

let addKnownCompoundClues = function (clueList: ClueList.Compound, clueCount: number, validateAll: boolean, fast: boolean): void {
    Assert(clueCount > 1);
    // this is currently only callable once per clueCount.
    Assert(!getKnownClueMap(clueCount));
    Assert(!getKnownSourceMap(clueCount));

    State.knownClueMapArray[clueCount] = {};
    State.knownSourceMapArray[clueCount] = {};

    clueList.forEach(clue => {
        if (clue.ignore) {
            return; // continue
        }
        clue.src = clue.src.split(',').sort().toString();
        let result = addCompoundClue(clue, clueCount, validateAll, fast);
        if (!State.ignoreLoadErrors) {
            if (!result.success) {
                console.error(`VALIDATE FAILED KNOWN COMPOUND CLUE: '${clue.src}':${clueCount}`);
            }
        }
        addKnownClue(clueCount, clue.name, clue.src);
    }, this);
};

//

let addKnownClue = function (count: number, name: string, source: string, nothrow: boolean = false): boolean {
    let clueMap = State.knownClueMapArray[count];
    if (!_.has(clueMap, name)) {
        log(`clueMap[${name}] = [${source}]`);
        clueMap[name] = [source];
    } else if (!clueMap[name].includes(source)) {
        log(`clueMap[${name}] += ${source} (${count})`);
        clueMap[name].push(source);
    } else {
        if (nothrow) return false;
        throw new Error('duplicate clue name/source' + 
                        '(' + count + ') ' +
                        name + ' : ' + source);
    }
    return true;
};

//

let removeKnownClue = function (count: number, name: string, source: string, nothrow: boolean): boolean {
    let clueMap = State.knownClueMapArray[count];
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
};

//

let saveClues = function (counts: number|number[]): void {
    if (_.isNumber(counts)) {
        counts = [counts] as number[];
    }
    counts = counts as number[];
    Debug(`saving clueLists ${counts}`);
    for (const count of counts) {
        saveClueList(State.clueListArray[count], count);
        Debug(`saved clueList ${count}, length: ${State.clueListArray[count].length}`);
    }
};

//

let addClue = function (count: number, clue: Clue.Compound, save = false, nothrow = false): boolean {
    clue.src = clue.src.split(',').sort().toString();
    if (addKnownClue(count, clue.name, clue.src, nothrow)) {
        (getClueList(count) as ClueList.Compound).push(clue);
        if (save) {
            saveClues(count);
        }
        return true;
    }
    return false;
};

//

let removeClue = function (count: number, clue: Clue.Compound, save = false, nothrow = false): boolean {
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (removeKnownClue(count, clue.name, clue.src, nothrow)) {
        _.remove(State.clueListArray[count], function (elem: Clue.Compound) {
            return (elem.name === clue.name) && (elem.src === clue.src);
        });
        if (save) {
            saveClues(count);
        }
        return true;
    }
    return false;
};

//

let addMaybe = function (name: string, srcNameList: string | string[], note: string | undefined, save = false): boolean {
    return false;
    /*
    if (typeof srcNameList === "string") {
        srcNameList = srcNameList.split(',');
    }
    //Expect(srcNameList).is.an.Array();
    let count = _.size(srcNameList);
    //Expect(count).is.above(1); // at.least(2);
    let clue: Clue.Type = {
        name: name,
        src: _.toString(srcNameList)
    };
    if (note) {
        clue.note = note;
    }
    this.maybeListArray[count].push(clue);
    if (save) {
        this.maybeListArray[count].save(this.getMaybeFilename(count));
    }
    return true;
    */
};

//

let addRejectCombos = function (clueList: ClueList.Compound, clueCount: number): void {
    clueList.forEach(clue => {
        let srcNameList = clue.src.split(',');
        if (_.size(srcNameList) !== clueCount) {
            log(`WARNING! reject word count mismatch` +
                     `, expected {clueCount}, actual ${_.size(srcNameList)}, ${srcNameList}`);
        }
        addRejectSource(srcNameList);
    });
};

//

let saveRejects = function (counts: number|number[]): void {
    if (_.isNumber(counts)) {
        counts = [counts] as number[];
    }
    counts = counts as number[];
    counts.forEach(count => {
        State.rejectListArray[count].save(getRejectFilename(count));
    });
};

//

let addReject = function (srcNameList: string|string[], save = false): boolean {
    if (_.isString(srcNameList)) {
        srcNameList = (srcNameList as string).split(',');
    }
    srcNameList = srcNameList as string[];
    let count = _.size(srcNameList);
    //Expect(count).is.above(1); // at.least(2);
    if (addRejectSource(srcNameList)) {
        State.rejectListArray[count].push({
            src: _.toString(srcNameList)
        });
        if (save) {
            State.rejectListArray[count].save(getRejectFilename(count));
        }
        return true;
    }
    return false;
};

//

let addRejectSource = function (srcNameList: string|string[]): boolean {
    if (_.isString(srcNameList)) {
        srcNameList = (srcNameList as string).split(',');
    }
    srcNameList = srcNameList as string[];
    //Expect(srcNameList).is.an.Array().and.not.empty();
    srcNameList.sort();
    log('addRejectSource: ' + srcNameList);

    if (isKnownSource(srcNameList.toString())) {
        console.log('WARNING! not rejecting known source, ' + srcNameList);
        return false;
    }
    if (isRejectSource(srcNameList)) {
        console.log('WARNING! duplicate reject source, ' + srcNameList);
        // i had this return false commented out for some reason,
        // but it should be here.
        return false;
    }
    State.rejectSourceMap[srcNameList.toString()] = true;
    return true;
};

// source is string containing sorted, comma-separated clues

let isKnownSource = function (source: string, count = 0): boolean {
    // check for supplied count
    if (count > 0) {
        return _.has(getKnownSourceMap(count), source);
    }
    // check for all counts
    return State.knownSourceMapArray.some(srcMap => _.has(srcMap, source));
};

// source: csv string or array of strings

let isRejectSource = function (source: string | string[]): boolean {
    return _.has(State.rejectSourceMap, source.toString());
};

//

export let getCountListForName = (name: string): CountList => {
    // TODO: filter/map
    let countList: CountList = [];
    for (const [index, clueMap] of State.knownClueMapArray.entries()) {
        if (_.has(clueMap, name)) {
            countList.push(index);
        }
    };
    return countList;
};

//
/*
let getSrcListMapForName = function (name: string): Record<number, string[]> {
    let srcListMap = {};
    // TODO: maxClues
    for (let index = 1; index < State.maxClues; ++index) {
        let srcList = State.knownClueMapArray[index][name];
        if (srcList) {
            srcListMap[index] = srcList;
        }
    }
    return srcListMap;
};
*/

// TODO this is failing with ncList.length > 1

let primaryNcListToNameSrcLists = function (ncList: NameCount.List): NameCount.List[] {
    let log = 0 && (ncList.length > 1); // nameSrcLists.length > 1) {
    let srcLists = ncList.map(nc => primaryNcToSrcList(nc));
    let indexLists = srcLists.map(srcList => [...Array(srcList.length).keys()]);  // e.g. [ [ 0 ], [ 0, 1 ], [ 0 ], [ 0 ] ]
    let nameSrcLists: NameCount.List[] = Peco.makeNew({
        listArray: indexLists,
        max:        999 // TODO technically, clue-types[variety].max_clues
    }).getCombinations()
        .map((indexList: number[]) => indexList.map((value, index) =>  // indexList -> primaryNameSrcList
            NameCount.makeNew(ncList[index].name, _.toNumber(srcLists[index][value]))))
        .filter((nameSrcList: NameCount.List) =>                             // filter out duplicate primary sources
            _.uniqBy(nameSrcList, NameCount.count).length === nameSrcList.length)
        .map((nameSrcList: NameCount.List) =>                                // sort by primary source (count).
            _.sortBy(nameSrcList, NameCount.count));       //   TODO: couldn't i just do forEach(list => _.sortBy(list, xx)) ? (is sortBy sort-in-place list list.sort()?)

    if (log) {
        console.log(`    ncList: ${ncList}`);
        console.log(`    nameSrcLists: ${nameSrcLists}`);
        console.log(`    uniq: ${_.uniqBy(nameSrcLists, _.toString)}`);
    }
    return _.uniqBy(nameSrcLists, _.toString);
};

//

let primaryNcListToNameSrcSets = function (ncList: NameCount.List): Set<string>[] {
    let log = 0 && (ncList.length > 1); // nameSrcLists.length > 1) {
    let srcLists = ncList.map(nc => primaryNcToSrcList(nc));
    let indexLists = srcLists.map(srcList => [...Array(srcList.length).keys()]);  // e.g. [ [ 0 ], [ 0, 1 ], [ 0 ], [ 0 ] ]
    let nameSrcSets: Set<string>[] = Peco.makeNew({
        listArray: indexLists,
        max:       2 // 999 // TODO? technically, clue-types[variety].max_clues
    }).getCombinations()
        .reduce((result: Set<string>[], indexList: number[]) => {
            let set = new Set<string>();
            //indexList.forEach((value, index) => set.add(NameCount.makeNew(ncList[index].name, srcLists[index][value])));
            indexList.forEach((value, index) => set.add(NameCount.makeCanonicalName(ncList[index].name, _.toNumber(srcLists[index][value]))));
            if (set.size === indexList.length) {
                result.push(set); // no duplicates
            }
            return result;
        }, []);

    if (log) {
        //console.log(`    ncList: ${ncList}`);
        //console.log(`    nameSrcLists: ${nameSrcLists}`);
        //console.log(`    uniq: ${_.uniqBy(nameSrcLists, _.toString)}`);
    }
    // TODO: put in some code to check for dupes, to see if we're actually seeing dupes. might slow down a little.
    
    //return _.uniqBy(nameSrcLists, _.toString);
    return nameSrcSets;
};

//

let primaryNcToSrcList = function (nc: NameCount.Type): string[] {
    if (nc.count !== 1) throw new Error(`nc.count must be 1 (${nc})`);
    const source = State.knownClueMapArray[1][nc.name];
    return (_.isArray(source) ? source : [source]) as string[];
};

//

let makeSrcNameListArray = function (nc: NameCount.Type): string[][] {
    let srcNameListArray: string[][] = [];
    getKnownClueMap(nc.count)[nc.name].forEach(src => {
        srcNameListArray.push(src.split(','));
    });
    return srcNameListArray;
};

// args:
//  sum:     args.sum,
//  max:     args.max,
//  require: args.require
//
// A "clueListArray" is an array where each element is a cluelist,
// such as [clues1, clues1, clues2].
//
// Given a sum, such as 3, generate an array of addend arrays that
// that add up to that sum, such as [ [1, 2], [2, 1] ], and return an
// array of clueListArrays of the corresponding clue counts, such
// as [ [clues1, clues2], [clues2, clues1] ].

/*
export let getClueSourceListArray = function (args: any): any {
    Log.info(`++clueSrcListArray, sum: ${args.sum}, max: ${args.max}, require: ${args.require}`);

    let clueCountListArray: CountList[] = Peco.makeNew({
        sum:     args.sum,
        max:     args.max,
        require: args.require
    }).getCombinations(); 

    Debug(`clueCountListArray(${clueCountListArray.length}): ${Stringify(clueCountListArray)}`);

    let clueSourceListArray: any[] = [];
    // TODO: .filter()
    clueCountListArray.forEach(clueCountList => {
        let clueSourceList: any[] = [];
        if (clueCountList.every(count => {
            // empty lists not allowed
            if (_.isEmpty(State.clueListArray[count])) {
                return false;
            }
            clueSourceList.push({ 
                list:  State.clueListArray[count],
                count
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
};
*/

//

/*
let filterAddends = function (addends: any, sizes: any): any[] {
    let filtered: any[] = [];
    addends.forEach(list => {
        if (sizes.every(size => {
            return list.indexOf(Number(size)) > -1;
        })) {
            filtered.push(list);
        }
    });
    return filtered;
};
*/

//

export let filter = function (srcCsvList: string[], clueCount: number, map: any = {}): any {
    let known = 0;
    let reject = 0;
    let duplicate = 0;
    srcCsvList.forEach(srcCsv => {
        if (isRejectSource(srcCsv)) {
            log(`isRejectSource(${clueCount}) ${srcCsv}`);
            ++reject;
        } else {
            if (isKnownSource(srcCsv, clueCount)) {
        log(`isKnownSource(${clueCount}) ${srcCsv}`);
        ++known;
        }
            if (_.has(map, srcCsv)) {
                log(`duplicate: ${srcCsv}`);
                ++duplicate;
            }
            map[srcCsv] = true;
        }
    });
    return { map, known, reject, duplicate };
};

// TODO: return type.  from Validator?
function singleEntry (nc: NameCount.Type, source: string): any {
    return {
    results: [
        {
        ncList: [nc],
        nameSrcList: [NameCount.makeNew(nc.name, _.toNumber(source))]
        }
    ]
    };
};

//

// returns:  array of SourceData or array of { entry: SourceData, sources: string }
// 

export let getKnownSourceMapEntries = function (nc: NameCount.Type, andSources = false) {
    const clueMap = State.knownClueMapArray[nc.count];
    if (!clueMap) throw new Error(`No clueMap at ${nc.count}`);
    const sourcesList = clueMap[nc.name];
    if (!sourcesList) throw new Error(`No sourcesList at ${nc.name}`);
    return sourcesList
        .map(sources => sources.split(',').sort().toString()) // sort sources
        .map((sources, index) => {
            const entry = (nc.count === 1)
                ? singleEntry(nc, sourcesList[index])
                : getKnownSourceMap(nc.count)[sources];
            //console.log(` sources: ${sources}`); // entry: ${Stringify(entry)}, entry2: ${Stringify(entry2)}`);
            return andSources ? { entry, sources } : entry;
        }); 
};

//

let getKnownClues = function (nameList: string|string[]): Record<string, Clue.Any[]> {
    if (_.isString(nameList)) {
        nameList = (nameList as string).split(',');
    }
    nameList = nameList as string[];
    const sourceCsv = nameList.sort().toString();
    let nameClueMap: Record<string, Clue.Any[]> = {};
    State.knownSourceMapArray.forEach(srcMap => {
        if (_.has(srcMap, sourceCsv)) {
            for (const clue of srcMap[sourceCsv].clues) {
                if (!_.has(nameClueMap, clue.name)) {
                    nameClueMap[clue.name] = [];
                }
                nameClueMap[clue.name].push(clue);
            }
        }
    });
    return nameClueMap;
};

//

let getKnownClueNames = function (nameList: string | string[]): string[] {
    return _.keys(getKnownClues(nameList));
};

//

let getClueCountListArray = function (nameList: string[]): CountList[] {
    // each count list contains the clueMapArray indexes in which
    // each name appears
    let countListArray: CountList[] = Array(_.size(nameList)).fill(0).map(_ => []);
    for (let count = 1; count <= State.maxClues; ++count) {
        let map = State.knownClueMapArray[count];
        //Expect(map).is.ok(); // I know this will fail when I move to synth clues
        nameList.forEach((name, index) => {
            if (_.has(map, name)) {
                countListArray[index].push(count);
            }
        });
    }
    return countListArray;
};

//

let getValidCounts = function (nameList: string[], countListArray: any): number[] {
    if ((nameList.length === 1) || isRejectSource(nameList)) return [];

    let addCountSet = new Set<number>();
    let known = false;
    let reject = false;
    Peco.makeNew({
        listArray: countListArray,
        max:       State.maxClues
    }).getCombinations()
        .forEach((clueCountList: number[]) => {
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
};

//

let getCountList = function (nameOrList: string|string[]): CountList {
    return _.isString(nameOrList)
        ? getCountListForName(nameOrList as string)
        : getValidCounts(nameOrList as string[], getClueCountListArray(nameOrList as string[]));
};

//
//
export let getPrimaryClue = function (nameSrc: NameCount.Type): Clue.Primary {
    const match = getClueList(1).find(clue => 
        clue.name === nameSrc.name && _.toNumber(clue.src) === nameSrc.count);
    if (!match) throw new Error(`can't find clue: ${nameSrc}`);
    return match as Clue.Primary;
};

//

let getPrimarySources = function (): string[] {
    let primarySources: string[] = [];
    let hash: any = {};
    for (const clue of getClueList(1)) {
        if (_.has(hash, clue.src)) continue;
        primarySources.push(clue.src);
        hash[clue.src] = true;
    }
    //console.log(`primarysources: ${primarySources}`);
    return primarySources;
};

//

export let getInversePrimarySources = function (sources: string[]): string[] {
    let inverseSources: string[] = [];
    for (const src of getPrimarySources()) {
        if (_.includes(sources, src)) continue;
        inverseSources.push(src);
    }
    return inverseSources;
};

//

let addClueForCounts = function (countSet: Set<number>, name: string, src: string,
                                 propertyName: string, options: any): number {
    const clue: Clue.Compound = { name, src };
    return Array.from(countSet)
        .reduce((added: number, count: number) => {
            if (!propertyName) {
                if (options.compound) {
                    let result = addCompoundClue(clue, count, true, true);
                    if (!result.success) throw new Error(`addCompoundclue failed, ${clue}:${count}`);
                }
                if (addClue(count, clue, options.save, true)) { // save, nothrow
                    console.log(`${count}: added ${name} as ${src}`);
                    added += 1;
                } else {
                    console.log(`${count}: ${name} already present`);
                }
            } else {
                const list = getClueList(count);
                // any because arbirtray property indexing
                let foundClue: any = _.find(list, clue);
                if (foundClue) {
                    foundClue[propertyName] = true;
                    console.log(`${count}: added '${propertyName}' property to ${name}:${src}`);
                    added += 1;
                }
            }
            return added;
        }, 0);
};

//

let removeClueForCounts = function (countSet: Set<number>, name: string, src: string,
                                    propertyName: string, options: any = {}): number {
    let removed = 0;
    for (let count of countSet.keys()) {
        if (!propertyName) {
            if (removeClue(count, { name, src }, options.save, options.nothrow)) {
                Debug(`removed ${name}:${count}`);
                removed += 1;
            } else {
                // not sure this should ever happen. removeClue throws atm.
                Debug(`${name}:${count} not present`);
            }
        } else {
            const list = getClueList(count);
            // any because arbirtray property indexing
            let foundClue: any = _.find(list, { name, src });
            if (foundClue) {
                delete foundClue[propertyName];
                console.log(`${count}: removed '${propertyName}' property from ${name}:${src}`);
            removed += 1;
            }
        }
    }
    return removed;
};

// each count list contains the clueMapArray indexes in which
// each name appears

let getKnownClueIndexLists = function (nameList: string[]): CountList[] {
    let countListArray: CountList[] = Array(_.size(nameList)).fill(0).map(_ => []);
    //Debug(countListArray);
    // TODO: is maxClues correct here
    for (let count = 1; count <= State.maxClues; ++count) {
        const map = State.knownClueMapArray[count];
        if (!_.isUndefined(map)) {
            nameList.forEach((name, index) => {
                if (_.has(map, name)) {
                    countListArray[index].push(count);
                }
            });
        } else {
            console.log('missing known cluemap #' + count);
        }
    }
    // verify that all names were found
    nameList.forEach((name, index) => {
        //Expect(countListArray[index]).is.ok();
    });
    return countListArray;
};

//
// args:
//   add=Name
//   remove=Name
//   reject
//   isKnown
//   isReject
//

export let addRemoveOrReject = function (args: any, nameList: string[],
                                         countSet: Set<number>, options: any = {}): number {
    let count = 0;
    nameList = nameList.sort();
    if (args.add) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --add due to single source');
        } else if (args.isReject) {
            console.log('WARNING! cannot add known clue: already rejected, ' + nameList);
        } else {
            count = addClueForCounts(countSet, args.add, nameList.toString(), args.property, options);
        }
    } else if (args.remove) {
        Debug(`remove [${args.remove}] as ${nameList} from ${[...countSet.values()]}`);
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --remove due to single source');
        } else {
            let removeOptions = { save: options.save, nothrow: true };
            count = removeClueForCounts(countSet, args.remove, nameList.toString(), args.property, removeOptions);
        }
    } else if (args.reject) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --reject due to single source');
        } else if (args.isKnown) {
            console.log('WARNING! cannot add reject clue: already known, ' + nameList);
        } else if (addReject(nameList.toString(), true)) {
            console.log('updated');
        }
        else {
            console.log('update failed');
        }
    }
    return count;
};

// TODO: List.isAnySublistEmpty
let isAnySubListEmpty = (listOfLists: any[][]): boolean => {
    let loggy = false;
    if (loggy) {
        console.log('listOfLists:');
        console.log(listOfLists);
    }
    for (const list of listOfLists) {
        if (loggy) {
            console.log('  list:');
            console.log(list);
        }
        if (_.isEmpty(list)) return true;
    }
    return false;
};

let getAllCountListCombosForNameList = function (nameList: string[], max = 999): CountList[] { // TODO technically, clue-types[variety].max_clues
    const countListArray = getKnownClueIndexLists(nameList);
    Debug(countListArray);
    if (isAnySubListEmpty(countListArray)) return [];
    return Peco.makeNew({
        listArray: countListArray,
        max
    }).getCombinations();
};

let buildNcListFromNameListAndCountList = function (nameList: string[], countList: number[]): NameCount.List {
    return countList.map((count, index) => NameCount.makeNew(nameList[index], count));
};

let buildNcListsFromNameListAndCountLists = function (nameList: string[], countLists: any[]): NameCount.List[] {
    let ncLists: NameCount.List[] = [];
    for (const countList of countLists) {
        ncLists.push(buildNcListFromNameListAndCountList(nameList, countList));
    }
    return ncLists;
};

export let buildNcListsFromNameList = function (nameList: string[]): NameCount.List[] {
    const countLists = getAllCountListCombosForNameList(nameList);
    if (_.isEmpty(countLists)) {
        Debug('empty countLists or sublist');
        return [];
    }
    Debug(countLists);
    return buildNcListsFromNameListAndCountLists(nameList, countLists);
};

let getListOfPrimaryNameSrcLists = function (ncList: NameCount.List): NameCount.List[] {
    // TODO: can use reduce() here too 
    let listOfPrimaryNameSrcLists: NameCount.List[] = [];
    //console.log(`ncList: ${ncList}`);
    for (const nc of ncList) {
        if (!_.isNumber(nc.count) || _.isNaN(nc.count)) throw new Error(`Not a valid nc: ${nc}`);
        //console.log(`  nc: ${nc}`);
        let lastIndex = -1;
        let entries;
        for (;;) {
            entries = getKnownSourceMapEntries(nc, true);
            if (!_.isArray(entries) || _.isEmpty(entries)) {
                console.log(`  explosion, nc: ${nc}, `); //entries: ${Stringify(entries)}`);
                process.exit(-1);
            }
            //console.log(`  entries: ${Stringify(entries)}`);
            if (nc.count === 1) {
                break; // TODO BUGG this might be wrong for multiple equivalent primary sources
            }
            
            let currIndex = -1;
            entries.every((item, index) => {
                if (item.entry) return true;
                currIndex = index;
                return false;
            });
            if (currIndex === -1) break;
            if (currIndex === lastIndex) {
                console.log(`currIndex == lastIndex (${currIndex})`);
                process.exit(-1);
            }
            
            // TODO BUGG skip this part for primary clues?
            const sources = entries[currIndex].sources;
            
            //console.log(`adding nc: ${nc}, sources ${sources}`); // entries: ${Stringify(entries)}`);
            
            const clue = { name: nc.name, src: sources };
            addCompoundClue(clue, nc.count, true);
            //
            // TODO
            //
            // call addClue here too
            //addClue(clue, nc.count)
            lastIndex = currIndex;
        }
        
        // verify that no other entries are undefined
        
        const primaryNameSrcLists = _.flatten(entries.map((item, index) => {
            const entry = item.entry;
            if (!entry || !entry.results || !_.isArray(entry.results) || _.isEmpty(entry.results)) {
                // || _.isEmpty(item.sources)) {
                
                console.log(`  explosion2, nc: ${nc}, sources: ${item.sources}`); //, entry: ${Stringify(entry)}, entries: ${Stringify(entries)}`);
                
                if (!entry) console.log('entry null');
                else if (!entry.results) console.log('entry.results null');
                else if (!_.isArray(entry.results)) console.log('entry.results not array');
                else if (_.isEmpty(entry.results)) console.log('entry.results empty');
                process.exit(-1);
            }
            return entry.results.map((result: ValidateResult) => result.nameSrcList);
        }));
        //primaryNameSrcLists.forEach(nameSrcList => console.log(`    nameSrcList: ${nameSrcList}`));
        listOfPrimaryNameSrcLists.push(primaryNameSrcLists);
    }
    return listOfPrimaryNameSrcLists;
};

export let origbuildListsOfPrimaryNameSrcLists = function (ncLists: NameCount.List[]): any[] {
    return ncLists.map(ncList => getListOfPrimaryNameSrcLists(ncList));
};

export let buildListsOfPrimaryNameSrcLists = function (ncLists: NameCount.List[]): any[] {
    return ncLists.map(ncList => {
    let result = getListOfPrimaryNameSrcLists(ncList);
    if (!result[0][0]) {
        throw new Error(`!result[0][0]: ${ncList}`);
    }
    return result;
    });
};

function getCompatiblePrimaryNameSrcList (listOfListOfPrimaryNameSrcLists: any[]) {
    //console.log(`${Stringify(listOfListOfPrimaryNameSrcLists)}`);
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists => [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length
    return Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) => sum + listOfNameSrcLists.length, 0)
    }).getCombinations()
        .some((comboList: number[]) => {
            const nameSrcList = comboList.reduce((nameSrcList, element, index) => {
                let nsList = listOfListOfPrimaryNameSrcLists[index][element];
                //console.log(`nameSrcList: ${nameSrcList}, element ${element}, index ${index}, nsList: ${nsList}`);
                if (nsList) {
                    nameSrcList.push(...nsList);
                } else {
                    console.log(`no nsList for index: ${index}, element: ${element}`);
                }
                return nameSrcList;
            }, [] as NameCount.List);
            const numUniq = _.uniqBy(nameSrcList, NameCount.count).length;
            return (numUniq === nameSrcList.length) ? nameSrcList : undefined;
        });
};

//
// So it seems i added a "fast" version of getCountListArrays specifically for the copy-from
// use case, that does not use Validator (and hence is faster).
// Since that time, I pretty significantly optimized the Validator. So maybe we can go back
// to using the old one (have to supply fast:true or default to fast).
// The problem with not using the old one, is I need to replicate enforcement of the
// "restrictToSameClueNumber" flag for primary NameSrcs. Not an unachievable task necessarily,
// but there is definitely some benefit to having that logic in only one spot.
// Or I suppose I could just throw a "validateSources()" call into the "fast" method and
// that might be a good compromise (have to supply fast:true or default to fast).
//

export let fast_getCountListArrays = function (nameCsv: string, options: any): any[] {
    const nameList = nameCsv.split(',').sort();
    Debug(`fast_getCountListArrays for ${nameList}`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    const ncLists = buildNcListsFromNameList(nameList);
    //console.log(`ncLists: ${ncLists}`);
    if (_.isEmpty(ncLists)) {
        console.log(`No ncLists for ${nameList}`);
        return [];
    }
    return buildListsOfPrimaryNameSrcLists(ncLists)
        .reduce((compatibleNcLists, listOfListOfPrimaryNameSrcLists, index) => {
            const compatibleNameSrcList = getCompatiblePrimaryNameSrcList(listOfListOfPrimaryNameSrcLists);
            console.log(`${ncLists[index]}  ${compatibleNameSrcList ? 'VALID' : 'invalid'}`);
            if (compatibleNameSrcList) {
                compatibleNcLists.push(ncLists[index]);
            }
            return compatibleNcLists;
        }, []);
};

// getCountListArrays
// 
// Probably not the most unique function name possible.
//
// Document what this does.
//

let invalidHash: any = {}; // hax

export let getCountListArrays = function (nameCsv: string, options: any): any {
    const validateAll = options.any ? false : true;
    const nameList = nameCsv.split(',').sort();
    Debug(`++getCountListArrays(${nameList})`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    const resultList = getAllCountListCombosForNameList(nameList);
    if (_.isEmpty(resultList)) {
        console.log(`No matches for ${nameList}`);
        return null;
    }

    let addRemoveSet;
    if (options.add || options.remove) {
        addRemoveSet = new Set();
    }
    let valid: any[] = [];
    let known: any[] = [];
    let rejects: any[] = [];
    let clues: any[] = [];
    let invalid: any[] = [];

    //console.log(`size ${resultList.length}`);

    for (const clueCountList of resultList) {
        const sum = clueCountList.reduce((a, b) => a + b);
        const start = new Date();
        let uniqueCounts = _.uniqBy(clueCountList, _.toNumber);
        if (0) {
            console.log(`${nameList}`);
            console.log(` sum: ${sum}, countList: ${clueCountList}, uniqueCounts: ${uniqueCounts}`);
        }
        let ncListStr = clueCountList.map((count, index) => NameCount.makeNew(nameList[index], count)).toString();
        let result: ValidateSourcesResult = invalidHash[ncListStr];
        if (!result) {
            result = Validator.validateSources({
                sum:         sum,
                nameList:    nameList,
                count:       nameList.length,
                require:     uniqueCounts,
                fast:        options.fast,
                validateAll
            });
        }
        invalidHash[ncListStr] = result;
        
        if (!result.success) {
            //console.log(`invalid: ${nameList}  CL ${clueCountList}  x ${x} sum ${sum}  validateAll=${validateAll}`);
            invalid.push(clueCountList);
        } else if (isRejectSource(nameList)) {
            rejects.push(clueCountList);
        } else if (nameList.length === 1) {
            let name = nameList[0];
            let srcList = State.clueListArray[sum]
                .filter(clue => clue.name === name)
                .map(clue => clue.src);
            if (srcList.length > 0) {
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
                clues.push({ countList: clueCountList, nameList: srcList });
            }
        } else {
            let any: SourceData = getKnownSourceMap(sum)[nameList.toString()];
            if (any) {
                known.push({
                    countList: clueCountList,
                    nameList: (any.clues as ClueList.Compound).map(clue => clue.name)
                });
            } else {
                valid.push(clueCountList);
            }
            addRemoveSet && addRemoveSet.add(sum);
        }
    }
    return { valid, known, rejects, invalid, clues, addRemoveSet };
};

// haxy.  Consider: Extends Clue.Compound + count: number
interface ClueIndex extends NameCount.Type {
    source: string;  // csv of names, or primary source #
}

interface RecursiveNode extends ClueIndex {
    recurse: boolean;
}

let getClue = function (clueIndex: ClueIndex): Clue.Any {
    const list = getClueList(clueIndex.count);
    // TODO:consider "restrict fields" to name/src if extend Clue.Compound
    let clue: Clue.Any = _.find(list, { name: clueIndex.name, src: clueIndex.source }) as Clue.Any;
    if (clue) return clue;
    
    console.error(`no clue for ${clueIndex.name}:${clueIndex.count} as ${clueIndex.source}(${typeof clueIndex.source})`);
    // TODO:consider "restrict fields" to name if extend Clue.Compound
    clue = _.find(list, { name: clueIndex.name }) as Clue.Any;
    if (clue) console.error(`  found ${clueIndex.name} in ${clueIndex.count} as ${Stringify(clue)}`);
    throw new Error(`no clue`);
};

//
// key types:
//   if value is non-array object value-type
// A.  if key has more than one subkey, source is sortedSubKeyNames
//   - else key has only one subkey,
//     - confirm subkey value type is array
// B.    if key is non-primary, source is splitSubkeyNames
// C.    else key is primary, sources is val[key][0].split(',')[1].toNumber();
//   else array value-type
// D.  array value type, add one node for each value[0].split(',') name:_.toNumber(src) entry, no subkeys

//{
// A.
//  'jack:3': {             // non-array value type, more than one subkey, source is sortedSubKeyNames, recurse
// B:
//    'card:2': {           // non-array value type, one subkey, key is non-primary, source is splitSubkeyNames, recurse
// D:
//  'bird:1,red:1': [   // array value type: add one node for each value[0].split(',') name:_.toNumber(src) entry, no recurse
//    'bird:2,red:8'
//  ]
//    },
// C:                       
//    'face:1': {           // non-array value type, one subkey, key is primary, sources is val[key][0].split(':')[1].toNumber(), no recurse
//  'face:1': [
//    'face:10'
//  ]
//    }
//  }
//}
//
//{
// D:
//  'face:1': [
//    'face:10'
//  ]
//}
//
export let recursiveGetCluePropertyCount = function (
    nc: NameCount.Type | null,
    resultMap: any, // object | string[] | NameCount.Type[] ? probably fix resultMap before attempting this.
    propertyName: Clue.PropertyName.Any,
    top: boolean = true): Clue.PropertyCounts.Type
{
    const loggy = 0;
    if (top && loggy) console.log(`${Stringify(resultMap)}`);

    const keys: string[] = _.keys(resultMap);
    let nodes: RecursiveNode[] = _.flatMap(keys, key => {
        // TODO: BUG:: key may be a ncCsv
        let val: any = resultMap[key];
        Assert(_.isObject(val), `Mystery key: ${key}`);
        // A,B,C: non-array object value type
        if (!_.isArray(val)) {
            let nc = NameCount.makeNew(key);
            let source: string;
            let subkeys: string[] = _.keys(val);
            let recurse = true;
            if (subkeys.length > 1) {
                // A: more than one subkey
                source = NameCount.nameListFromStrList(subkeys).sort().toString();
            } else { // only one subkey
                if (nc.count !== 1) {
                    // B. non-primary key
                    source = NameCount.nameListFromCsv(subkeys[0]).toString();
                } else {
                    // C. primary key
                    // first array element
                    // ugly. TS requiring it. typify resultmap to fix.
                    let ncList= (val as Record<string, NameCount.List>)[key];
                    source = `${ncList[0].count}`;
                    recurse = false;
                }
            }
            return { name: nc.name, count: nc.count, source, recurse };
        } else {
            // D: array value type 
            // first array element
            let csv = val[0].toString();
            let list = NameCount.makeListFromCsv(csv);
            return list.map(sourceNc => {
                return {
                    name: sourceNc.name,
                    count: 1,
                    source: `${sourceNc.count}`,
                    recurse: false
                };
            });
        }
    });
    // in the case of e.g., -t night, the above code only considers the components of night,
    // but not night:N itself. that's represented by 'nc' here. nc.count *should never* be 1,
    // as it's unnecessary to call this function if it is.
    if (nc) {
        Assert(nc.count !== 1);
        nodes.push({
            name: nc.name,
            count: nc.count,
            source: NameCount.nameListFromStrList(keys).sort().toString(),
            recurse: false
        });
    }
    let counts = Clue.PropertyCounts.empty();
    for (let node of nodes) {
        if (loggy) console.log(Stringify(node));
        const clue = getClue(node);
        // primary clues only: add pre-computed propertyCount data to total, if it exists
        if (Clue.isPrimary(clue)) {
            clue.propertyCounts && Clue.PropertyCounts.add(counts, clue.propertyCounts[propertyName]);
        } else {
            Assert(node.count > 1);
            // compound clue: there is no propertyCount data, so just add the count (1 or 0)
            // of the properties on the clue itself
            Clue.PropertyCounts.add(counts, Clue.PropertyCounts.getCounts(clue, propertyName));
        }
        if (loggy) {
            if (clue[propertyName]) {
                console.log(`^^${propertyName}! total${counts.total} primary(${counts.primary}`);
            } else {
                console.log(`${Clue.toJSON(clue)}, clue[${propertyName}]=${clue[propertyName]}`);
            }
        }
        if (node.recurse) {
            let val = resultMap[NameCount.makeCanonicalName(node.name, node.count)];
            Clue.PropertyCounts.add(counts, recursiveGetCluePropertyCount(null, val, propertyName, false));
        }
    }
    return counts;
};
