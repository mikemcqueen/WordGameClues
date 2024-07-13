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

import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as CountBits from '../types/count-bits-fastbitset';
import * as MinMax from '../types/min-max';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';
import * as Source from './source';

const DATA_DIR     =  Path.normalize(`${Path.dirname(module.filename)}/../../../data/`);
const REJECTS_DIR  = 'rejects';

type CountList = number[];

export type SourceMapValue = {
    clues: ClueList.Any;
};

type ClueMap = Record<string, string[]>;
type SourceMap = Record<string, SourceMapValue>;

export type AllCandidates = Sentence.CandidatesContainer[];
export interface AllCandidatesContainer {
    allCandidates: AllCandidates;     // one per sentence (if parsed as a "sentence")
}

type InternalStateBase = {
    clueListArray: ClueList.Any[];    // the JSON "known" clue files in an array
    knownClueMapArray: ClueMap[];     // map clue name to list of clue sourceCsvs
    variations: Sentence.Variations;  // "global" variations aggregated from all sentences
    sentences: Sentence.Type[];

    uniquePrimaryClueNames: string[];

    dir: string;

    ignoreLoadErrors: boolean;
    loaded: boolean;
    logging: boolean;
    logLevel: number;
    
    maxClues: number;
}

type InternalState = InternalStateBase & AllCandidatesContainer;

const initialState = (): InternalState => {
    return {
        clueListArray: [],
        knownClueMapArray: [],

        sentences: [],
        variations: Sentence.emptyVariations(),

        uniquePrimaryClueNames: [],

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
    if (nc.count === 1) {
        return anyCandidateHasClueName(nc.name);
    }
    else if (_.has(getKnownClueMap(nc.count), nc.name)) {
        return true;
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
        // prob do this elsewhere
        //Native.addUniquePrimaryClueNameVariations(variations[key]);
    }
};

const addNamesForVariationName = (variation_name: string,
    variations: Sentence.VariationMap, names: Set<string>): void =>
{
    for (let key of Object.keys(variations)) {
        for (let name of variations[key]) {
            if (name === variation_name) {
                names.add(key);
                break;
            }
        }
    }
}

// this is kinda awkward and slow, but is only used for -t during output
// phase so shouldn't matter too much. don't use it though.
export const getAllNamesForVariationName = (name: string): Set<string> => {
    let names = new Set<string>();
    for (let sentence of State.sentences) {
        if (!sentence) continue;
        addNamesForVariationName(name, sentence.anagrams, names);
        addNamesForVariationName(name, sentence.synonyms, names);
        addNamesForVariationName(name, sentence.homophones, names);
    }
    return names;
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
        const keys = _.keys(container.nameIndicesMap);
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
    Native.setPrimaryNameSrcIndicesMap(_.keys(nameSourcesMap), values_lists(nameSourcesMap));
};

const primaryClueListPostProcessing = (args: any): void => {
    // TODO: got this backwards
    const sentences: Sentence.Type[] = State.sentences;

    // Add all variations for all sentences before building candidates.
    let variations = Sentence.emptyVariations();
    for (let sentence of sentences) {
        if (!sentence) continue;
        Sentence.addAllVariations(sentence, variations);
    }

    // 2nd pass through sentences
    // Generate unique *component* names (no name variations)
    // Build and populate State.allCandidates using name variations
    let uniqueComponentNames = new Set<string>();
    for (let i = 1; i < sentences.length; ++i) {
        const sentence = sentences[i];
        if (args.verbose) {
            // just info, but i want it output to stderr
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
    // Required by combo-maker first/next. It could also just use C++ addon
    // functions, or eventually be converted to C++ entirely.
    State.uniquePrimaryClueNames = initUniquePrimaryClueNames(uniqueComponentNames,
        args.addVariations ? sentences : undefined);
    State.variations = variations;
    State.sentences = sentences;
    // Call addKnownPrimaryClues() last, after State.allCandidates is populated
    addKnownPrimaryClues();
    showMinMaxLengths();
    cacheAllPrimaryClueSources();
};

let num_validates = 0;
let validate_duration = 0;

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
        Native.setCompoundClueNameSourcesMap(count, _.entries(getKnownClueMap(count)));
    }
    const t_dur = new Duration(t0, new Date()).milliseconds;
    if (args.verbose) {
        console.error(`addCompound max(${args.max_sources})  - ${PrettyMs(t_dur)}` +
            `, num_validates(${num_validates}) - ${PrettyMs(validate_duration)}`);
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

// Should rename this, as it doesn't actually add a clue.
// validateCompoundClueAndUpdateState()
// 
let addCompoundClue = (clue: Clue.Compound, count: number, args: any): boolean => {
    let nameList = clue.src.split(',').sort();
    let srcCsv = nameList.toString();
    let vs_result = true;
    // new sources need to be validated
    if (!Native.isKnownSourceMapEntry(count, srcCsv)) {
        const t0 = new Date();
        vs_result = Native.validateSources(clue.name, nameList, count, args.validateAll);
        validate_duration += new Duration(t0, new Date()).milliseconds;
        num_validates++;
    }
    if (vs_result && args.validateAll) {
        Native.addCompoundClue({name: clue.name, count}, srcCsv);
    }
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
    Assert(!getKnownClueMap(clueCount));

    State.knownClueMapArray[clueCount] = {};

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

export let getCountListForName = (name: string): CountList => {
    let countList: CountList = [...Array(State.knownClueMapArray.length).keys()];
    countList = countList.filter((count: number) => {
        return count && _.has(getKnownClueMap(count), name);
    });
    if (_.isEmpty(countList) || (countList[0] !== 1)) {
        if (anyCandidateHasClueName(name)) {
            countList = [1, ...countList];
        }
    }
    return countList;
};

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
        /*
        if (isRejectSource(srcCsv)) {
            //log(`isRejectSource(${clueCount}) ${srcCsv}`);
            ++result.reject;
        } else {
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
        //}
    });
    return result;
};

const addClueForCounts = (counts: number[], name: string, src: string,
    propertyName: string, options: any): number =>
{
    const clue: Clue.Compound = { name, src };
    return counts
        .filter((count: number) => { return !options.addMax || (count <= options.addMax); })
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

const removeClueForCounts = (counts: number[], name: string, src: string,
    propertyName: string, options: any = {}): number =>
{
    let removed = 0;
    for (let count of counts) {
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
    counts: number[], options: any = {}): number =>
{
    let count = 0;
    nameList = nameList.sort();
    if (args.add) {
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --add due to single source');
        }
        else {
            count = addClueForCounts(counts, args.add, nameList.toString(), args.property, options);
        }
    } else if (args.remove) {
        Debug(`remove [${args.remove}] as ${nameList} from ${counts}`);
        if (nameList.length === 1) {
            console.log('WARNING! ignoring --remove due to single source');
        } else {
            let removeOptions = {
                save: options.save,
                nothrow: true
            };
            count = removeClueForCounts(counts, args.remove, nameList.toString(), args.property, removeOptions);
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

