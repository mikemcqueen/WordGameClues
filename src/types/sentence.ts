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

//
//
interface SentenceBase {
    num: number;
    text: string;
    combinations: string[];
    components: Record<string, string[]>; 
}

export interface Variations {
    anagrams: Record<string, string[]>;
    synonyms: Record<string, string[]>;
    homophones: Record<string, string[]>;
}

/*
export interface VariationsContainer {
    variations: Variations;
}
*/

type StringToNumbersMap = {
    [key: string]: Set<number>;
};

export type NameSourcesMap = StringToNumbersMap;

// run-time only, not part of schema
export interface Candidate {
    clues: ClueList.Primary;
    nameSourcesMap: NameSourcesMap;
}

export type NameIndicesMap = StringToNumbersMap;

export interface CandidatesContainer {
    candidates: Candidate[];
    nameIndicesMap: NameIndicesMap;
}

export type Type = SentenceBase & Variations; // Container; // & CandidatesContainer;
export type List = Type[];

export let load = (dir: string, num: number): Type => {
    return makeFrom(getFilename(dir, num));
};

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

export const copyCandidatesContainer = (container: CandidatesContainer): CandidatesContainer => {
    return {
	candidates: copyCandidates(container.candidates),
	nameIndicesMap: copyStringToNumbersMap(container.nameIndicesMap)
    };
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

const validate = (sentence: Type): boolean => {
    return validateCombinations(sentence)
	&& validateVariations(sentence);
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

export const addAllVariations = (sentence: Type, variations: Variations): void => {
    addVariations(sentence.anagrams, variations.anagrams);
    addVariations(sentence.synonyms, variations.synonyms);
    addVariations(sentence.homophones, variations.homophones);
};

const addVariations = (fromVariations: Record<string, string[]>,
    toVariations: Record<string, string[]>): void =>
{
    for (let key of Object.keys(fromVariations)) {
	if (_.has(toVariations, key)) {
	    // this could be more relaxed; i could compare values to ensure equality
	    throw new Error(`duplicate variation: ${key}`)
	}
	const wordList = fromVariations[key];
	toVariations[key] = wordList;
	for (let word of wordList) {
	    if (_.has(toVariations, word)) {
		// this could be more relaxed; i could compare values to ensure equality
		throw new Error(`duplicate variation ${key}: ${word}`)
	    }
	    toVariations[word] = wordList;
	}
    }
};

export const buildAllCandidates = (sentence: Type, variations: Variations):
    CandidatesContainer =>
{
    let candidates: Candidate[] = [];
    let src = 1_000_000 * sentence.num; // up to 10000 variations of up to 100 names
    const sortedText = stripAndSort(sentence.text);
    for (const combo of sentence.combinations) {
	//console.error(`combo: ${combo}`);
	const nameListMap = buildCandidateNameListMap(combo.split(' '), sentence.components);
	for (let nameList of nameListMap.values()) {
	    //console.error(`nameList: ${nameList}`);
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
	nameIndicesMap: buildNameIndicesMap(candidates, variations)
    };
};

// TODO: alien technology. revisit or ignore at your peril.
// pretty sure this has some profound wrongness about it.
//
const buildCandidateNameListMap = (componentList: string[],
    components: Record<string, string[]>, startIndex = 0,
    results = new Map<string, string[]>()): Map<string, string[]> => 
{
    const log = false;
    if (log) {
	console.error(`  IN: ${componentList} @ ${componentList[startIndex]}` +
	    ` (${startIndex} of ${componentList.length})`);
    }
    for (let i = startIndex; i < componentList.length; ++i) {
	const component = componentList[i];
	const replace = _.has(components, component);
	const alternates: string[] = replace ? components[component] : [component];
	for (let alternate of alternates) {
	    let copy = componentList.slice(0, i);
	    let offset = 0;
	    const split = alternate.split(' ');
	    if (split.length > 1) {
		copy.push(...split);
	    } else {
		copy.push(alternate);
		offset = 1;
	    }
	    for (let j = i + 1; j < componentList.length; ++j) {
		copy.push(componentList[j]);
	    }
	    const nextIndex = i + offset;
	    if (/*(startIndex < componentList.length - 1) || */(nextIndex < copy.length)) {
		buildCandidateNameListMap(copy, components, nextIndex, results);
	    } else if ((startIndex == componentList.length - 1) && (nextIndex === copy.length)) {
		const key = copy.slice().sort().join('');
		if (!results.has(key)) {
		    results.set(key, copy);
		    if (log) {
			console.error(`  OUT: ${copy} @ start(${componentList[startIndex]}), next(${copy[nextIndex]})` +
			    `, startIndex ${startIndex} of ${componentList.length}, nextIndex ${nextIndex} of ${copy.length}`);
		    }
		}
	    } else {
		if (log) {
		    console.error(`  skip: ${copy} @ start(${componentList[startIndex]}), next(${copy[nextIndex]})` +
			`, startIndex ${startIndex} of ${componentList.length}, nextIndex ${nextIndex} of ${copy.length}`);
		}
	    }
	}
    }
    return results;
};

const buildClueList = (num: number, nameList: string[], src: number):
    ClueList.Primary =>
{
    let clues: ClueList.Primary = [];
    for (let name of nameList) {
	clues.push({
	    num,
	    name,
	    src: `${src}`
	});
	src += 1;
    }
    return clues;
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
	const alternateNames = getAlternateNames(clue.name, variations);
	for (let altName of alternateNames) {
	    if (!_.has(map, altName)) {
		map[altName] = set;
	    } else {
		// if this fires, we've got mismatched name/variation somewhere
		Assert(map[altName] === set);
	    }
	}
    }
    return map;
};

const buildNameIndicesMap = (candidates: Candidate[], variations: Variations):
    NameIndicesMap => 
{
    let map: NameIndicesMap = {};
    for (let i = 0; i < candidates.length; ++i) {
	const candidate = candidates[i];
	const keys = Object.keys(candidate.nameSourcesMap);
	for (let name of keys) {
	    if (!_.has(map, name)) {
		map[name] = new Set<number>();
	    }
	    let set = map[name];
	    set.add(i);
	    const alternateNames = getAlternateNames(name, variations);
	    for (let altName of alternateNames) {
		if (!_.has(map, altName)) {
		    map[altName] = set;
		} else {
		    // if this fires, we've got mismatched name/variation somewhere
		    Assert(map[altName] === set);
		}
	    }
	}
    }
    return map;
};

const getAlternateNames = (name: string, variations: Variations): string[] => {
    let result: string[] = [];
    result.push(...(variations.anagrams[name] || []));
    result.push(...(variations.synonyms[name] || []));
    result.push(...(variations.homophones[name] || []));
    return result;
};

export const getUsedSources = (nameSrcList: NameCount.List): number[] => {
    let result: number[] = [];
    nameSrcList.filter(nameSrc => isCandidateSource(nameSrc.count))
	.forEach(nameSrc => {
	    const [sentenceIndex, variationIndex] = getSourceIndices(nameSrc.count);
	    // this should never fire. just being defensive.
	    if (result[sentenceIndex] !== undefined) {
		if (result[sentenceIndex] !== variationIndex) {
		    //console.error(NameCount.listToString(nameSrcList));
		    throw new Error(`oopsie ${NameCount.toString(nameSrc)}`);
		}
	    } else {
		result[sentenceIndex] = variationIndex;
	    }
	});
    return result;
}

export const isCandidateSource = (src: number): boolean => {
    return src >= 1_000_000;
}

export const getSourceIndices = (src: number): [number, number] => {
    Assert(isCandidateSource(src));
    return [Math.floor(src / 1_000_000), Math.floor((src % 1_000_000) / 100)];
}

export const legacySrcList = (nameSrcList: NameCount.List): number[] => {
    return nameSrcList.map(nc => nc.count).filter(src => (src < 1_000_000));
}

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
