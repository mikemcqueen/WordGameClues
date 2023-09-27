//
// clue-manager.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Log            = require('../../modules/log')('clue-manager');
const Peco           = require('../../modules/peco');
const Clues          = require('../../modules/clue-types');

const Assert         = require('assert');
const Debug          = require('debug')('clue-manager');
const Duration       = require('duration');
const Native     = require('../../../build/experiment.node');
const Path       = require('path');
const PrettyMs   = require('pretty-ms');
const Stringify2 = require('stringify-object');
const stringify  = require('javascript-stringify').stringify;
const Validator  = require('./validator');

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as CountBits from '../types/count-bits-fastbitset';
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';
import * as Source from './source';
import type { ValidateResult, ValidateSourcesResult } from './validator';

const DATA_DIR     =  Path.normalize(`${Path.dirname(module.filename)}/../../../data/`);
const REJECTS_DIR  = 'rejects';

type CountList = number[];

export type SourceMapValue = {
    clues: ClueList.Any;
//    results: ValidateResult[];
};

type NcResultData = {
    list: ValidateResult[];
    set: Set<string>;
};

type ClueMap = Record<string, string[]>;
type SourceMap = Record<string, SourceMapValue>;
type NcResultMap = Record<string, NcResultData>;

export type AllCandidates = Sentence.CandidatesContainer[];
export interface AllCandidatesContainer {
    allCandidates: AllCandidates;     // one per sentence (if parsed as a "sentence")
}

type InternalStateBase = {
    clueListArray: ClueList.Any[];    // the JSON "known" clue files in an array
    knownClueMapArray: ClueMap[];     // map clue name to clue src
    knownSourceMapArray: SourceMap[]; // map sourceCsv to SourceMapValue
    ncResultMaps: NcResultMap[];      // map known NCs to result list

    variations: Sentence.Variations;  // "global" variations aggregated from all sentences
    sentences: Sentence.Type[];

    uniquePrimaryClueNames: string[];

    maybeListArray: any[];            // the JSON maybe files in an array
    rejectListArray: any[];           // the JSON reject files in an array
    rejectSourceMap: any;             // map reject source to true/false (currently)
    
    dir: string;

    ignoreLoadErrors: boolean;
    loaded: boolean;
    logging: boolean;
    logLevel: number;
    
    maxClues: number;
//    numPrimarySources: number;
}

type InternalState = InternalStateBase & AllCandidatesContainer;

const initialState = (): InternalState => {
    return {
        clueListArray: [],
        knownClueMapArray: [],
        knownSourceMapArray: [],
        ncResultMaps: [],

        sentences: [],
        variations: Sentence.emptyVariations(),

        uniquePrimaryClueNames: [],

        // TODO: remove? maybe, at least
        maybeListArray: [],
        rejectListArray: [],
        rejectSourceMap: {},

        dir: '',
        
        ignoreLoadErrors: false,
        loaded: false,
        logging: false,
        logLevel: 0,
        
        maxClues: 0,
        allCandidates: []
    };
};

let State: InternalState = initialState();

export let isLoaded = (): boolean => { return State.loaded; }
export function getMaxClues (): number { return State.maxClues; }
export function setLogging (onOff: boolean): void { State.logging = onOff; }

export function getClueList (index: number): ClueList.Any {// Compound {
    Assert(index > 1, "cluelist(1) is bogus");
    return State.clueListArray[index];
}

export function getKnownClueMap (count: number): ClueMap {
    return State.knownClueMapArray[count];
}

export function getKnownSourceMap (count: number): SourceMap {
    return State.knownSourceMapArray[count];
}

export function getNcResultMap (count: number): NcResultMap {
    return State.ncResultMaps[count];
}

export function getAllCandidates (): AllCandidates {
    return State.allCandidates;
}

export const getVariations = (): Sentence.Variations => {
    return State.variations;
};

export const getSentence = (num: number): Sentence.Type => {
    return State.sentences[num];
};

export const getCandidatesContainer = (sentence: number):
    Sentence.CandidatesContainer =>
{
    return State.allCandidates[sentence];
};

export const getUniqueClueNameCount = (clueCount: number) => {
    if (clueCount === 1) {
        return State.uniquePrimaryClueNames.length;
    }
    const list = getClueList(clueCount);
    //if (!list) console.error(`no cluelist @ ${clueCount}`);
    // for now. could do better
    return (list && list.length) || 0;
};

export const getUniqueClueName = (clueCount: number, nameIndex: number) => {
    if (clueCount === 1) {
        return State.uniquePrimaryClueNames[nameIndex];
    }
    // for now. could do better
    return getClueList(clueCount)[nameIndex].name;
};

//

function Stringify (val: any): string {
    return stringify(val, (value: any, indent: number, stringify: any) => {
        if (typeof value == 'function') return "function";
        return stringify(value);
    }, " ");
}

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

const log = (text: string): void => {
    let pad = '';
    let index;
    if (State.logging) {
        for (let index = 0; index < State.logLevel; index += 1) { pad += ' '; }
        console.log(pad + text);
    }
};

const anyCandidateHasClueName = (name: string, allCandidates: AllCandidates
    = State.allCandidates): boolean =>
{
    for (let container of allCandidates) {
        if (!container) continue;
        const indices = container.nameIndicesMap[name] || [];
        for (let index of indices) {
            if (container.candidates[index]) {
                return true;
            }
        }
    }
    return false;
};

export const isKnownNc = (nc: NameCount.Type): boolean => {
    // for legacy primary, and all compound clues
    if (_.has(getKnownClueMap(nc.count), nc.name)) {
        return true;
    }
    // special case for candidate primary clues in sentences
    if (nc.count === 1) {
        return anyCandidateHasClueName(nc.name);
    }
    return false;
};

interface SaveClueListOptions {
    dir : string;
}

export const saveClueList = (list: ClueList.Compound, count: number,
    options?: SaveClueListOptions): void =>
{
    ClueList.save(list, getKnownFilename(count, options?.dir));
};

//////////

const getKnownFilename = function (count: number, dir?: string): string {
    return Path.format({
        dir: !dir ? State.dir : `${DATA_DIR}${dir}`,
        base: `clues${count}.json`
    });
};

interface LoadClueListOptions {
    dir : string;
}

export const loadClueList = (count: number,
    options?: LoadClueListOptions): ClueList.Any =>
{
    const filename = getKnownFilename(count, options?.dir);
    return ClueList.makeFrom({ filename, primary: count === 1 });
};

const loadSentence = (num: number, args: any): number => {
    if (args.verbose) {
        console.error(`loading sentence ${num}`);
    }
    let maxClues = 0;// TODO
    State.sentences[num] = Sentence.load(State.dir, num);
    return maxClues;
};

const autoSource = (clueList: ClueList.Primary, args: any): void => {
    for (let clue of clueList) {
        if (clue.num) {
            loadSentence(_.toNumber(clue.num), args);
        }
    }
};

const addPrimaryNameSrcToMap = (name: string, src: string): void => {
    const clueMap = getKnownClueMap(1);
    if (!_.has(clueMap, name)) {
        clueMap[name] = [];
    }
    clueMap[name].push(src);
};

const addPrimaryClueAndVariationsToMap = (clue: Clue.Primary,
    variations: Sentence.Variations): void => 
{
    addPrimaryNameSrcToMap(clue.name, clue.src);
    const nameVariations = Sentence.getNameVariations(clue.name, variations);
    for (let nameVariation of nameVariations) {
        addPrimaryNameSrcToMap(nameVariation, clue.src);
    }
};

const addKnownPrimaryClues = (): void => {
    State.knownClueMapArray[1] = {};
    for (let container of getAllCandidates()) {
        if (!container) continue;
        for (let candidate of container.candidates) {
            for (let clue of candidate.clues) {
                const sentenceNum = Source.getCandidateSentence(Number(clue.src));
                const sentence = getSentence(sentenceNum);
                addPrimaryClueAndVariationsToMap(clue, sentence);
            }
        }
    }
};

const addUniqueName = (toList: string[], name: string, hash: Set<string>): void => {
    if (!hash.has(name)) {
        hash.add(name);
        toList.push(name);
    }
};

const addUniqueVariationNames = (toList: string[],
    variations: Sentence.VariationMap, hash: Set<string>): void =>
{
    for (let key of Object.keys(variations)) {
        for (let name of variations[key]) {
            addUniqueName(toList, name, hash);
        }
    }
};

const initUniquePrimaryClueNames = (uniqueComponentNames: Set<string>,
    sentences?: Sentence.Type[]): string[] =>
{
    let result: string[] = [];
    let hash = new Set<string>();
    // add unique component names from sentences
    for (let name of uniqueComponentNames.values()) {
        addUniqueName(result, name, hash);
    }
    // add variation names
    // This is necessary because of complicated reasons. show-components uses
    // ComboMaker's PreCompute(). PreCompute uses first/next() iterators to
    // populate the "sourceListMap" that is passed on to the C++ addon code
    // for actually generating combos. We don't want to add these in that case,
    // (I think), because it's much faster to generate combos with a smaller
    // input list, and just add additional combos with substituted variation
    // names after results are computed.
    // In the case of show-components though, we need these added. Otherwise
    // it's impossible to -t <anagram/synonym/homonym>.
    if (sentences) {
        for (let sentence of sentences) {
            if (!sentence) continue;
            addUniqueVariationNames(result, sentence.anagrams, hash);
            addUniqueVariationNames(result, sentence.synonyms, hash);
            addUniqueVariationNames(result, sentence.homophones, hash);
        }
    }
    return result;
};

const showMinMaxLengths = (): void => {
    let minMaxTotal = MinMax.init(0);
    let minMaxSentence = MinMax.init();
    for (let container of getAllCandidates()) {
        if (!container) continue;
        MinMax.update(minMaxSentence, container.minMaxLengths.max);
        MinMax.add(minMaxTotal, container.minMaxLengths);
    }
    console.error(`clues: min(${minMaxTotal.min}), max(${minMaxTotal.max})` +
        `, longest(${minMaxSentence.max})`);
};

const buildPrimaryClueNameSourcesMap = (allCandidates = getAllCandidates()):
    Sentence.NameSourcesMap =>
{
    let nameSourcesMap: Sentence.NameSourcesMap = {};
    // add candidates clue sources
    for (let container of allCandidates) {
        if (!container) continue;
        let keys = _.keys(container.nameIndicesMap);
        for (let name of keys) {
            if (!nameSourcesMap.hasOwnProperty(name)) {
                nameSourcesMap[name] = new Set<number>();
            }
            let sources = nameSourcesMap[name];
            // add all sources for name to sources list
            const indices = container.nameIndicesMap[name];
            for (let index of indices) {
                const candidate = container.candidates[index];
                Assert(candidate && _.has(candidate.nameSourcesMap, name));
                const candidate_sources = candidate.nameSourcesMap[name];
                Assert(!_.isEmpty(candidate_sources));
                for (let src of candidate_sources) {
                    sources.add(src);
                }
            }
        }
    }
    return nameSourcesMap;
};

const values_lists = (nameSourcesMap: Sentence.NameSourcesMap): number[][] => {
    return _.values(nameSourcesMap).map(src_set => [...src_set.values()]);
};

const cacheAllPrimaryClueSources = (): void => {
    let nameSourcesMap = buildPrimaryClueNameSourcesMap();
    Native.setPrimaryClueNameSourcesMap(_.keys(nameSourcesMap), values_lists(nameSourcesMap));
};

const primaryClueListPostProcessing = (args: any): void => {
    // TODO: got this backwards
    const sentences: Sentence.Type[] = State.sentences;

    // Add all variations for all sentences before building candidates.
    let variations = Sentence.emptyVariations();
    for (let sentence of sentences) {
        if (!sentence) continue;
        Sentence.addAllVariations(variations, sentence);
    }

    // 2nd pass through sentences
    // Generate unique *component* names (no name variations)
    // Build and populate State.allCandidates using name variations
    let uniqueComponentNames = new Set<string>();
    for (let i = 1; i < sentences.length; ++i) {
        const sentence = sentences[i];
        // just info, but i want it output to stderr
        if (args.verbose) {
            console.error(`sentence ${i}:`);
        }
        Assert(sentence, `sentence ${i}:`);
        const names = Sentence.getUniqueComponentNames(sentence);
        names.forEach(name => uniqueComponentNames.add(name));
        const container = Sentence.buildAllCandidates(sentence, variations, args);
        State.allCandidates[i] = container;
        if (args.verbose) {
            console.error(` names: ${names.size}, variations: ${container.candidates.length}`);
        }
    }
    State.uniquePrimaryClueNames = initUniquePrimaryClueNames(uniqueComponentNames,
        args.addVariations ? sentences : undefined);
    State.variations = variations;
    State.sentences = sentences;
    // Call addKnownPrimaryClues() last, after State.allCandidates is populated
    addKnownPrimaryClues();
    showMinMaxLengths();
    cacheAllPrimaryClueSources();
};

let total_sources = 0;

// args:
//  baseDir:  base directory (meta, synth)
//  ignoreErrors:
//  validateAll:
//
export const loadAllClues = function (args: any): void {
    State.dir = Clues.getDirectory(args.clues);
    if (args.ignoreErrors) {
        State.ignoreLoadErrors = true;
    }
    State.maxClues = args.max;

    let primaryClueList: ClueList.Primary = loadClueList(1) as ClueList.Primary;
    if (primaryClueList[0].src !== 'auto') {
        throw new Error('something something src="auto"');
    }
    autoSource(primaryClueList, args);
    // if using -t, add primary variations to uniqueNames
    primaryClueListPostProcessing(args);
    const t0 = new Date();
    for (let count = 2; count <= args.max_sources; ++count) {
        let clueList: ClueList.Compound = loadClueList(count);
        State.clueListArray[count] = clueList;
        addKnownCompoundClues(clueList, count, args);
        Native.setCompoundClueNames(count, _.keys(getKnownClueMap(count)));
    }
    const t_dur = new Duration(t0, new Date()).milliseconds;
    if (args.verbose) {
        console.error(`addCompound max(${args.max_sources})` +
            `, total_sources(${total_sources}) - ${PrettyMs(t_dur)}`);
    }
    State.loaded = true;
};

const getRejectFilename = function (count: number): string {
    return Path.format({
        dir:  `${DATA_DIR}${REJECTS_DIR}`,
        base: `rejects${count}.json`
    });
};

const findConflicts = (set: Set<string>, nameSrcList: NameCount.List) : boolean => {
    for (let ncCsv of set.values()) {
        const ncList = NameCount.makeListFromCsv(ncCsv);
        Assert(ncList.length === nameSrcList.length);
        let conflicts: string[][] = []; // array of 2-tuples
        for (let i = 0; i < ncList.length; ++i) {
            if (ncList[i].name != nameSrcList[i].name) {
                conflicts.push([ncList[i].name, nameSrcList[i].name]);
            }
        }
        // length 1 is always bad. *probably* same with 3 but not proven.
        // lengths 2 and 4 are recoverable. only works for 2 for now (lazy).
        // ex: [ 'low' (13), 'owl' (70) ], [ 'owl' (13), 'low' (70) ]
        if ((conflicts.length != 2) ||
            (conflicts[0][0] != conflicts[1][1]) ||
            (conflicts[0][1] != conflicts[1][0]))
        {
            console.error(`conflicts(${conflicts.length}): ${conflicts[0]}, ${conflicts[1]}`);
            return true;
        }
    }
    return false;
};

// TODO: return dstList, spread push @ caller
// well, that might be tricky because I initialize map from dstList.
//
let appendUniqueResults = (ncResult: NcResultData, ncStr: string, 
    fromResults: ValidateResult[], args: any) : void => 
{
    for (let result of fromResults) {
        const key = NameCount.listToCountList(result.nameSrcList).sort().join(',');
        if (ncResult.set.has(key)) {
            continue;
        }
        ncResult.set.add(key);
        ncResult.list.push(result);
    }
};

const addResultsToNcResultMap = (results: ValidateResult[], name: string,
    count: number, args: any) =>
{
    let ncResultMap = getNcResultMap(count);
    if (!ncResultMap) {
        ncResultMap = State.ncResultMaps[count] = {};
    }
    const ncStr = NameCount.makeCanonicalName(name, count);
    if (!ncResultMap[ncStr]) {
        ncResultMap[ncStr] = {
            list: [],
            set: new Set<string>()
        };
    }
    appendUniqueResults(ncResultMap[ncStr], ncStr, results, args);
};

// Should rename this, as it doesn't actually add a clue.
// validateCompoundClueAndUpdateState()
// 
let addCompoundClue = (clue: Clue.Compound, count: number, args: any): boolean => {
    let nameList = clue.src.split(',').sort();
    let srcMap = getKnownSourceMap(count);
    let srcKey = nameList.toString();

    //console.log(`${srcKey}:${count}`);

    // new sources need to be validated
    //    let vsResult : ValidateSourcesResult = { success: true };
    let vs_result = true;
    //if (!_.has(srcMap, srcKey)) {
    if (!Native.isKnownSourceMapEntry(count, srcKey)) {
        Debug(`## validating Known compound clue: ${srcKey}:${count}`);
        /*
        vsResult = Validator.validateSources(clue.name, {
            sum: count,
            nameList,
            count: nameList.length,
            fast: args.fast,
            validateAll: args.validateAll
        });
        */
        vs_result = Native.validateSources(clue.name, nameList, count,
            args.validateAll);

        // this is where the magic happens
        if (vs_result && args.validateAll) {
            //total_sources += vsResult.list!.length;
            srcMap[srcKey] = {
                clues: [],
            };
        }
    }
    /*
    else if (args.validateAll) {
        vsResult.list = srcMap[srcKey].results;
    }
    */
    if (vs_result && args.validateAll) {
        //addResultsToNcResultMap(vsResult.list!, clue.name, count, args);
        //Native.appendNcResults({ name: clue.name, count }, vsResult.list!);
        Native.appendNcResultsFromSourceMap({name : clue.name, count}, srcKey);
        (srcMap[srcKey].clues as ClueList.Compound).push(clue);
    }
    // NOTE: added above, commented below
    // TODO: I don't understand why I'm doing this in failure case.
    // should probably be inside above if block. maybe need to split
    // out validateAll as well, i'm not sure what the "not validateAll"
    // use case is anymore. (or what .clues is use for, for that matter).
    // The question is, is this appropriate for successful but non
    // validate-all scenarios, such as copy-from?  maybe.
    //(srcMap[srcKey].clues as ClueList.Compound).push(clue);
    return vs_result;
};

const addKnownClue = (count: number, name: string, source: string,
    nothrow: boolean = false): boolean =>
{
    let clueMap = getKnownClueMap(count);
    if (!_.has(clueMap, name)) {
        log(`clueMap[${name}] = [${source}]`);
        clueMap[name] = [source];
    } else if (!clueMap[name].includes(source)) {
        log(`clueMap[${name}] += ${source} (${count})`);
        clueMap[name].push(source);
    } else {
        if (nothrow) return false;
        throw new Error(`duplicate clue name/source (${count}) ${name}:${source}`);
    }
    return true;
};

const addKnownCompoundClues = (clueList: ClueList.Compound, clueCount: number,
    args: any): void =>
{
    Assert(clueCount > 1);
    // this is currently only callable once per clueCount.
    Assert(!getKnownClueMap(clueCount) && !getKnownSourceMap(clueCount));

    State.knownClueMapArray[clueCount] = {};
    State.knownSourceMapArray[clueCount] = {};

    clueList
        .filter(clue => !clue.ignore)
        .forEach(clue => {
            clue.src = clue.src.split(',').sort().toString();
            if (addCompoundClue(clue, clueCount, args)) {
                addKnownClue(clueCount, clue.name, clue.src);
            } else if (args.removeAllInvalid) {
                removeClue(clueCount, clue, true, false, true); // save, nothrow, force
            } else if (!State.ignoreLoadErrors) {
                console.error(`VALIDATE FAILED KNOWN COMPOUND CLUE:` +
                    ` '${clue.src}':${clueCount}, -t ${clue.src} --remove ${clue.name}`);
            }
        });
};

let removeKnownClue = function (count: number, name: string, source: string, nothrow: boolean): boolean {
    let clueMap = getKnownClueMap(count);
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

let saveClues = function (counts: number|number[]): void {
    if (_.isNumber(counts)) {
        counts = [counts] as number[];
    }
    counts = counts as number[];
    Debug(`saving clueLists ${counts}`);
    for (const count of counts) {
        saveClueList(getClueList(count), count);
        Debug(`saved clueList ${count}, length: ${State.clueListArray[count].length}`);
    }
};

let addClue = (count: number, clue: Clue.Compound, save = false, nothrow = false):
    boolean =>
{
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

let removeClue = function (count: number, clue: Clue.Compound,
    save = false, nothrow = false, force = false): boolean
{
    // sort src
    clue.src = clue.src.split(',').sort().toString();
    if (force || removeKnownClue(count, clue.name, clue.src, nothrow)) {
        _.remove(getClueList(count), function (elem: Clue.Compound) {
            return (elem.name === clue.name) && (elem.src === clue.src);
        });
        if (save) {
            saveClues(count);
        }
        return true;
    }
    return false;
};

let addReject = function (srcNameList: string|string[], save = false): boolean {
    if (_.isString(srcNameList)) {
        srcNameList = (srcNameList as string).split(',');
    }
    srcNameList = srcNameList as string[];
    let count = _.size(srcNameList);
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

let addRejectSource = function (srcNameList: string|string[]): boolean {
    if (_.isString(srcNameList)) {
        srcNameList = (srcNameList as string).split(',');
    }
    srcNameList = srcNameList as string[];
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

export let getCountListForName = (name: string): CountList => {
    let countList: CountList = [...Array(State.knownClueMapArray.length).keys()]
        .filter(count => count && _.has(getKnownClueMap(count), name));
    if (_.isEmpty(countList) || (countList[0] !== 1)) {
        if (anyCandidateHasClueName(name)) {
            countList = [1, ...countList];
        }
    }
    return countList;
};

/*
let primaryNcToSrcList = function (nc: NameCount.Type): string[] {
    if (nc.count !== 1) throw new Error(`nc.count must be 1 (${nc})`);
    const source = State.knownClueMapArray[1][nc.name];
    return (_.isArray(source) ? source : [source]) as string[];
};

// TODO this is failing with ncList.length > 1
let primaryNcListToNameSrcLists = function (ncList: NameCount.List): NameCount.List[] {
    let log = 0 && (ncList.length > 1); // nameSrcLists.length > 1) {
    let srcLists = ncList.map(nc => primaryNcToSrcList(nc));
    let indexLists = srcLists.map(srcList => [...Array(srcList.length).keys()]);  // e.g. [ [ 0 ], [ 0, 1 ], [ 0 ], [ 0 ] ]
    let nameSrcLists: NameCount.List[] = Peco.makeNew({
        listArray: indexLists,
        max:        999 // TODO technically, clue-types[variety].max_clues
    }).getCombinations()
        // map indexList -> primaryNameSrcList
        .map((indexList: number[]) => indexList.map((value, index) =>
            NameCount.makeNew(ncList[index].name, _.toNumber(srcLists[index][value]))))
        // filter out duplicate primary sources
        .filter((nameSrcList: NameCount.List) =>
            _.uniqBy(nameSrcList, NameCount.count).length === nameSrcList.length)
        // sort by primary source (count).
        //  TODO: couldn't i just do forEach(list => _.sortBy(list, xx)) ?
        // (is sortBy sort-in-place list list.sort()?)
        .map((nameSrcList: NameCount.List) =>
            _.sortBy(nameSrcList, NameCount.count));

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
            indexList.forEach((value, index) =>
                set.add(NameCount.makeCanonicalName(ncList[index].name,
                  _.toNumber(srcLists[index][value]))));
            if (set.size === indexList.length) {
                result.push(set); // no duplicates
            }
            return result;
        }, []);
    // TODO: put in some code to check for dupes, to see if we're actually seeing
    //       dupes. might slow down a little.
    //return _.uniqBy(nameSrcLists, _.toString);
    return nameSrcSets;
};

let makeSrcNameListArray = function (nc: NameCount.Type): string[][] {
    let srcNameListArray: string[][] = [];
    getKnownClueMap(nc.count)[nc.name].forEach(src => {
        srcNameListArray.push(src.split(','));
    });
    return srcNameListArray;
};
*/

export interface FilterResult {
    map: any;
    known: number;
    reject: number;
    duplicate: number;
}

export const emptyFilterResult = (): FilterResult => {
    return {
        map: {},
        known: 0,
        reject: 0,
        duplicate: 0
    };
};

export const filter = (srcCsvList: string[], clueCount: number, result: FilterResult):
    FilterResult =>
{
    srcCsvList.forEach(srcCsv => {
        if (isRejectSource(srcCsv)) {
            //log(`isRejectSource(${clueCount}) ${srcCsv}`);
            ++result.reject;
        } else {
            /*
            if (isKnownSource(srcCsv, clueCount)) {
                log(`isKnownSource(${clueCount}) ${srcCsv}`);
                ++result.known;
            }
            if (_.has(result.map, srcCsv)) {
                log(`duplicate: ${srcCsv}`);
                ++result.duplicate;
            }
            */
            result.map[srcCsv] = true;
        }
    });
    return result;
};

// TODO: return type.
const singleEntry = (nc: NameCount.Type, source: string): SourceMapValue => {
    let nameSrcList: NameCount.List = [NameCount.makeNew(nc.name, Number(source))];
    return {
        clues: [],
/* TODO?
        results: [            
            {
                ncList: [nc],
                nameSrcList,
                usedSources: Source.getUsedSources(nameSrcList),
                // TODO: this shouldn't be here.
                resultMap: undefined
            }
        ]
*/
    };
};

// returns:
// array of SourceMapValue
//   --or--
// array of { entry: SourceMapValue, sources: string }
// 
export const getKnownSourceMapEntries = (nc: NameCount.Type,
    andSources = false, args: any = {}): any[] =>
{
    Assert(!"getKnownSourceMapEntries() broken");
    return [];
/*
    const clueMap = getKnownClueMap(nc.count);
    if (!clueMap) throw new Error(`No clueMap at ${nc.count}`);
    const sourcesList = clueMap[nc.name];
    if (!sourcesList) {
        if (!args.ignoreErrors) {
            throw new Error(`No sourcesList at ${nc.name}:${nc.count}`);
        }
        if (!args.quiet) {
            console.error(`No sourcesList at ${nc.name}:${nc.count}`);
        }
        return [];
    }
    return sourcesList
        .map(sources => sources.split(',').sort().toString()) // sort sources
        .map((sources, index) => {
            const entry = (nc.count === 1)
                ? singleEntry(nc, sourcesList[index])
                : getKnownSourceMap(nc.count)[sources];
            return andSources ? { entry, sources } : entry;
        }); 
*/
};

//

const getKnownClues = (nameList: string|string[]): Record<string, Clue.Any[]> => {
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

export const getKnownClueNames = (nameList: string | string[]): string[] => {
    return Object.keys(getKnownClues(nameList));
};

const addClueForCounts = (countSet: Set<number>, name: string, src: string,
    propertyName: string, options: any): number =>
{
    const clue: Clue.Compound = { name, src };
    return Array.from(countSet)
        .filter((count: number) => { return !options.max || (count <= options.max); })
        .reduce((added: number, count: number) => {
            if (!propertyName) {
                if (options.compound) {
                    if (!addCompoundClue(clue, count, { validateAll: true, fast: true })) {
                        throw new Error(`addCompoundClue failed, ${clue}:${count}`);
                    }
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

const removeClueForCounts = (countSet: Set<number>, name: string, src: string,
    propertyName: string, options: any = {}): number =>
{
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
const getKnownClueIndexLists = (nameList: string[]): CountList[] => {
    let countListArray: CountList[] = Array(_.size(nameList)).fill(0).map(_ => []);
    // TODO: is maxClues correct here
    for (let count = 1; count <= getMaxClues(); ++count) {
        const map = getKnownClueMap(count);
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
export const addRemoveOrReject = (args: any, nameList: string[],
    countSet: Set<number>, options: any = {}): number =>
{
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
const isAnySubListEmpty = (listOfLists: any[][]): boolean => {
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

 // TODO technically, clue-types[variety].max_clues
const getAllCountListCombosForNameList = (nameList: string[], max = 999): CountList[] => {
    const countListArray = getKnownClueIndexLists(nameList);
    Debug(countListArray);
    if (isAnySubListEmpty(countListArray)) return [];
    return Peco.makeNew({
        listArray: countListArray
    }).getCombinations();
};

const buildNcListFromNameListAndCountList = (nameList: string[],
    countList: number[]): NameCount.List =>
{
    return countList.map((count, index) => NameCount.makeNew(nameList[index], count));
};

const buildNcListsFromNameListAndCountLists = (nameList: string[],
    countLists: any[]): NameCount.List[] =>
{
    let ncLists: NameCount.List[] = [];
    for (const countList of countLists) {
        ncLists.push(buildNcListFromNameListAndCountList(nameList, countList));
    }
    return ncLists;
};

export const buildNcListsFromNameList = (nameList: string[]):
    NameCount.List[] =>
{
    const countLists = getAllCountListCombosForNameList(nameList);
    if (_.isEmpty(countLists)) {
        Debug('empty countLists or sublist');
        return [];
    }
    Debug(countLists);
    return buildNcListsFromNameListAndCountLists(nameList, countLists);
};

const getListOfPrimaryNameSrcLists = (ncList: NameCount.List):
    NameCount.List[] =>
{
    // TODO: can use reduce() here too 
    let listOfPrimaryNameSrcLists: NameCount.List[] = [];
    for (const nc of ncList) {
        Assert(!_.isNaN(nc.count), `${nc}`); // this might be dumb. can a number be NaN?
        let lastIndex = -1;
        let entries;
        for (;;) {
            entries = getKnownSourceMapEntries(nc, true);
            Assert(_.isArray(entries) && !_.isEmpty(entries), `${entries}`);
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
            Assert(currIndex !== lastIndex, `${currIndex}`);

            // TODO BUGG skip this part for primary clues?
            const sources = entries[currIndex].sources;
            const clue = { name: nc.name, src: sources };
            //addCompoundClue(clue, nc.count, { validateAll: true });
            //
            // TODO
            //
            // call addClue here too
            //addClue(clue, nc.count)
            lastIndex = currIndex;
        }
        // verify that no other entries are undefined
        const primaryNameSrcLists = entries.flatMap((item, index) => {
            const entry = item.entry;
            Assert(entry && entry.results && _.isArray(entry.results) &&
                _.isEmpty(entry.results), `${entry}`);
            //unused: && !_.isEmpty(item.sources));
            return entry.results.map((result: ValidateResult) => result.nameSrcList);
        });
        listOfPrimaryNameSrcLists.push(primaryNameSrcLists);
    }
    return listOfPrimaryNameSrcLists;
};

export const origbuildListsOfPrimaryNameSrcLists =(ncLists: NameCount.List[]):
    any[] =>
{
    return ncLists.map(ncList => getListOfPrimaryNameSrcLists(ncList));
};

export const buildListsOfPrimaryNameSrcLists = (ncLists: NameCount.List[]):
    any[] =>
{
    return ncLists.map(ncList => {
        let result = getListOfPrimaryNameSrcLists(ncList);
        if (!result[0][0]) {
            throw new Error(`!result[0][0]: ${ncList}`);
        }
        return result;
    });
};

const getCompatiblePrimaryNameSrcList = (listOfListOfPrimaryNameSrcLists: any[]):
    any =>
{
    //console.log(`${Stringify(listOfListOfPrimaryNameSrcLists)}`);
    const listArray = listOfListOfPrimaryNameSrcLists.map(listOfNameSrcLists =>
        [...Array(listOfNameSrcLists.length).keys()]); // 0..nameSrcList.length
    return Peco.makeNew({
        listArray,
        max: listOfListOfPrimaryNameSrcLists.reduce((sum, listOfNameSrcLists) => sum + listOfNameSrcLists.length, 0)
    }).getCombinations().some((comboList: number[]) => {
        const nameSrcList = comboList.reduce((nameSrcList, element, index) => {
            let nsList = listOfListOfPrimaryNameSrcLists[index][element];
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
// There uused to be a problem with not using the old one, in that I needed to replicate
// enforcement of the "restrictToSameClueNumber" flag for primary NameSrcs. That no longer
// holds as we've no longer got legacy clues.
// Or I suppose I could just throw a "validateSources()" call into the "fast" method and
// that might be a good compromise (have to supply fast:true or default to fast).
//
export const fast_getCountListArrays = (nameCsv: string, options: any): any[] => {
    const nameList = nameCsv.split(',').sort();
    Debug(`fast_getCountListArrays for ${nameList}`);

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    const ncLists = buildNcListsFromNameList(nameList);
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
// TODO: unused. and, maybe never need to look at this again. but consider
// replacing with version implemented in show-components if so.
// there's also a fast_ version, which may not need to exist.

let invalidHash: any = {}; // hax

export const getCountListArrays = (nameCsv: string, options: any): any => {
    Assert(0 && "don't use this function i think");
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
        let uniqueCounts = _.uniqBy(clueCountList, _.toNumber); // or just _.uniq ?
        let ncListStr = clueCountList.map((count, index) => NameCount.makeNew(nameList[index], count)).toString();
        let result: ValidateSourcesResult = invalidHash[ncListStr];
        if (!result) {
            result = Validator.validateSources(undefined, {
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
            let any: SourceMapValue = getKnownSourceMap(sum)[nameList.toString()];
            if (any) {
                known.push({
                    countList: clueCountList,
                    nameList: (any.clues as ClueList.Compound).map(clue => clue.name)
                });
            } else {
                valid.push(clueCountList);
            }
            if (addRemoveSet) addRemoveSet.add(sum);
        }
    }
    return { valid, known, rejects, invalid, clues, addRemoveSet };
};
