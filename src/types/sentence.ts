//
// sentence.ts
//

'use strict';

import _ from 'lodash';
const Assert = require('assert');
const Fs = require('fs-extra');
const Path = require('path');
const Stringify = require('stringify-object');

import * as Clue from './clue';
import * as ClueList from './clue-list';
import * as NameCount from './name-count';
import * as Source from '../modules/source';

//////////

export type VariationMap = Record<string, string[]>;

interface SentenceBase {
    num: number;
    text: string;
    combinations: string[];
    components: VariationMap;
}

export interface Variations {
    anagrams: VariationMap;
    synonyms: VariationMap;
    homophones: VariationMap;
}
export type Type = SentenceBase & Variations;
export type List = Type[];

type StringToNumbersMap = {
    [key: string]: Set<number>;
};

type NameSourcesMap = StringToNumbersMap;

// run-time only, not part of schema
interface Candidate {
    clues: ClueList.Primary;
    nameSourcesMap: NameSourcesMap;
}

export type NameIndicesMap = StringToNumbersMap;

export interface CandidatesContainer {
    candidates: Candidate[];
    // Map a clue "name" to all of the candidates (via indices) that contain it
    nameIndicesMap: NameIndicesMap;
}

//////////

export const emptyVariations = (): Variations => {
    return {
        anagrams: {},
        synonyms: {},
        homophones: {}
    };
};

// maybe not needed
/*
const emptyCandidatesContainer = (): CandidatesContainer => {
    return {
        candidates: {},
        nameCandidatesMap: {}
    }
};
*/

const copyStringToNumbersMap = (fromMap: StringToNumbersMap): StringToNumbersMap => {
    let toMap = {};
    for (let key of Object.keys(fromMap)) {
        toMap[key] = new Set(fromMap[key]);
    }
    return toMap;
};

const copyCandidates = (srcCandidates: Candidate[]): Candidate[] => {
    let candidates = [...srcCandidates];
    for (let candidate of candidates) {
        candidate.nameSourcesMap = copyStringToNumbersMap(candidate.nameSourcesMap);
    }
    return candidates;
};

export const copyCandidatesContainer = (container: CandidatesContainer):
    CandidatesContainer =>
{
    return {
        candidates: copyCandidates(container.candidates),
        nameIndicesMap: copyStringToNumbersMap(container.nameIndicesMap)
    };
};

export const getCandidateSourcesForName = (container: CandidatesContainer,
    name: string): number[] =>
{
    const sources: number[] = [];
    const indices = container.nameIndicesMap[name] || [];
    for (let index of indices) {
        sources.push(...container.candidates[index].nameSourcesMap[name]);
    }
    return sources;
};

//////////

// "strip" spaces from string, sort resulting letters
const stripAndSort = (text: string): string => {
    return joinAndSort(text.split(' '));
};

// join strings, sort resulting letters
const joinAndSort = (arr: string[]): string => {
    return sortString(arr.join(''));
};

// sort string
const sortString = (str: string): string => {
    return str.split('').sort().join('');
};

// TODO: all three of these loops below could be 1 function that takes a strip/sort
// function param
const validateCombinations = (sentence: Type): boolean => {
    const sortedText = stripAndSort(sentence.text);
    for (let combo of sentence.combinations) {
        if (sortedText != stripAndSort(combo)) {
            console.error(`sentences (${sentence.num}) combination:` +
                ` ${sentence.text} != ${combo}`);
            return false;
        }
    }
    return true;
};

const validateVariations = (sentence: Type): boolean => {
    for (let key of Object.keys(sentence.components)) {
        const sortedText = stripAndSort(key);
        for (let component of sentence.components[key]) {
            if (sortedText != stripAndSort(component)) {
                console.error(`sentence (${sentence.num}) component:` +
                    ` ${key} != ${component}`);
                return false;
            }
        }
    }
    for (let key of Object.keys(sentence.anagrams)) {
        const sortedText = sortString(key);
        for (let anagram of sentence.anagrams[key]) {
            if (sortedText != sortString(anagram)) {
                console.error(`sentence (${sentence.num}) anagram:` +
                    ` ${key} != ${anagram}`);
                return false;
            }
        }
    }
    if (!sentence.anagrams) {
        // TODO: confirm they are actually anagrams
        console.error(`sentence ${sentence.num} missing 'anagrams'`);
        return false;
    }
    if (!sentence.synonyms) {
        console.error(`sentence ${sentence.num} missing 'synonyms'`);
        return false;
    }
    if (!sentence.homophones) {
        console.error(`sentence ${sentence.num} missing 'homophones'`);
        return false;
    }
    return true;
};

const validate = (sentence: Type): boolean => {
    return validateCombinations(sentence)
        && validateVariations(sentence);
};

const makeFrom = (filename: string): Type => {
    let sentence: Type;
    try {
        const json = Fs.readFileSync(filename, 'utf8');
        sentence = JSON.parse(json);
        if (!validate(sentence)) {
            //console.error(validate.errors);
            throw new Error(`invalid json`);
        }
    }
    catch(e) {
        throw new Error(`${filename}, ${e}`);
    }
    return sentence;
};

const getFilename = (dir: string, count: number): string => {
    return Path.format({ dir, base: `sentence${count}.json` });
};

export let load = (dir: string, num: number): Type => {
    return makeFrom(getFilename(dir, num));
};

//////////

// TODO: assumes sorted order. could add to set instead.
const listsEqual = (a: string[], b: string[]): boolean => {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

const addVariations = (toVariations: VariationMap,
    fromVariations: VariationMap): void =>
{
    for (let key of Object.keys(fromVariations)) {
        if (toVariations.hasOwnProperty(key)) {
            if (!listsEqual(toVariations[key], fromVariations[key])) {
                // this could be more relaxed; listsEqual assumes sorted
                throw new Error(`incompatible variation: ${key} (could be relaxed)`);
            }
            continue;
        }
        const wordList = fromVariations[key];
        toVariations[key] = wordList;
        for (let word of wordList) {
            if (_.has(toVariations, word)) {
                throw new Error(`duplicate variation ${key}: ${word}`);
            }
            toVariations[word] = wordList;
        }
    }
};

export const addAllVariations = (variations: Variations,
    sentence: Type): void =>
{
    addVariations(variations.anagrams, sentence.anagrams);
    addVariations(variations.synonyms, sentence.synonyms);
    addVariations(variations.homophones, sentence.homophones);
};

// domestically engineered version
// TODO: this could return a set. or an array. like [...result.values()]
//       the map is useful during construction but unnecessary for the
//       consumer.
const buildCandidateNameListMap = (combinationWords: string[],
    components: VariationMap, startIndex = 0,
    results = new Map<string, string[]>()): Map<string, string[]> => 
{
    const log = false;
    if (log) {
        console.error(`  IN: ${combinationWords} @ ${combinationWords[startIndex]}` +
            ` (${startIndex} of ${combinationWords.length})`);
    }
    let add = true;
    for (let i = startIndex; i < combinationWords.length; ++i) {
        const component = combinationWords[i];
        if (components.hasOwnProperty(component)) {
            // this word has a matching entry in the component map. for each alternate
            // component, fudge the alternate component word(s) and any remaining
            // combination words into a new combinationWords array and call recursively
            const alternates: string[] = components[component];
            for (let alternate of alternates) {
                let newCombinationWords = [
                    ...combinationWords.slice(0, i),
                    ...alternate.split(' '),
                    ...combinationWords.slice(i + 1, combinationWords.length)
                ];
                // if one of the alternate words is the same as the component, skip
                // over it so we don't end up in an infinite recursive loop
                let skip = (alternate === component) ? 1 : 0;
                buildCandidateNameListMap(newCombinationWords, components, i + skip, results);
            }
            add = false;
            break;
        }
    }
    if (add) {
        const key = combinationWords.slice().sort().join('');
        if (!results.has(key)) {
            results.set(key, combinationWords);
        }
    }
    return results;
};

const buildClueList = (num: number, nameList: string[], src: number):
    ClueList.Primary =>
{
    let clues: ClueList.Primary = [];
    for (let name of nameList) {
        clues.push({ num, name, src: `${src}` });
        src += 1;
    }
    return clues;
};

export const getNameVariations = (name: string, variations: Variations): string[] => {
    let names: string[] = [];
    names.push(...(variations.anagrams[name] || []));
    names.push(...(variations.synonyms[name] || []));
    names.push(...(variations.homophones[name] || []));
    let hash = new Set<string>(names);
    return [...hash.values()];
};

const buildNameSourcesMap = (clueList: ClueList.Primary, variations: Variations):
    NameSourcesMap =>
{
    let map: NameSourcesMap = {};
    for (let clue of clueList) {
        if (!_.has(map, clue.name)) {
            map[clue.name] = new Set<number>();
        }
        let set = map[clue.name];
        set.add(Number(clue.src));
        getNameVariations(clue.name, variations).forEach(name => {
            if (!_.has(map, name)) {
                map[name] = set;
            } else if (map[name] !== set) {
                // if this fires, we've got mismatched name/variation somewhere
                console.error(`sentence ${Source.getCandidateSentence(Number(clue.src))}` +
                    `, variation '${name}' of component '${clue.name}`);
                console.error(`  ${clue.name}: [${[...set.values()]}]`);
                console.error(`  ${name}: [${[...map[name].values()]}]`);
                Assert(false);
            }
        });
    }
    return map;
};

// The nameIndicesMap maps a "name" to all of the candidates (via indices)
// that contain that name.
//
const buildNameIndicesMap = (candidates: Candidate[]): NameIndicesMap => 
{
    let map: NameIndicesMap = {};
    for (let i = 0; i < candidates.length; ++i) {
        const candidate = candidates[i];
        const names = Object.keys(candidate.nameSourcesMap);
        for (let name of names) {
            // TODO: this is not ideal. we should be able to reuse the same set
            // specific for a "component", with all variation names.
            // It's probably not a big deal.
            if (!_.has(map, name)) {
                map[name] = new Set<number>();
            }
            map[name].add(i);
            // Ok so pretty sure this isn't needed because all variation names
            // were already added to nameSourcesMap.
            /*
            const nameVariations = getNameVariations(name, variations);
            for (let altName of nameVariations) {
                if (!_.has(map, altName)) {
                    map[altName] = set;
                } else {
                    // if this fires, we've got mismatched name/variation somewhere
                    Assert(map[altName] === set);
                }
            }
            */
        }
    }
    return map;
};

export const buildAllCandidates = (sentence: Type, variations: Variations):
    CandidatesContainer =>
{
    let log = true;
    let candidates: Candidate[] = [];
    let src = 1_000_000 * sentence.num; // up to 10000 variations of up to 100 names
    // TODO: similar logic to getUniqueComponentNames() which is unfortunate
    const sortedText = stripAndSort(sentence.text);
    for (const combo of sentence.combinations) {
        const nameListMap = buildCandidateNameListMap(combo.split(' '), sentence.components);
        if (log) {
            console.error(`nameListMap(${nameListMap.size}) :`);
            for (let list of nameListMap.values()) {
                console.error(`  ${list}`);
            }
        }
        for (let nameList of nameListMap.values()) {
            if (sortedText !== joinAndSort(nameList)) {
                throw new Error(`sentence '${sentence.text}' != nameList '${nameList}'`);
            }
            const clues = buildClueList(sentence.num, nameList, src);
            candidates.push({
                clues,
                nameSourcesMap: buildNameSourcesMap(clues, variations)
            });
            src += 100;
        }
    }
    return {
        candidates,
        nameIndicesMap: buildNameIndicesMap(candidates)
    };
};

export const getUniqueComponentNames = (sentence: Type): Set<string> => {
    let result = new Set<string>();
    // TODO: similar logic to buildAllCandidates() which is unfortunate
    for (const combo of sentence.combinations) {
        const nameListMap = buildCandidateNameListMap(combo.split(' '),
            sentence.components);
        for (let nameList of nameListMap.values()) {
            nameList.forEach(name => result.add(name));
        }
    }
    return result;
}

export const legacySrcList = (nameSrcList: NameCount.List): number[] => {
    return nameSrcList.map(nc => nc.count).filter(src => !Source.isCandidate(src));
}

