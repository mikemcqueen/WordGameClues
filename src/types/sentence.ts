//
// sentence.ts
//

'use strict';

import _ from 'lodash';
const Fs = require('fs-extra');
const Path = require('path');

import * as ClueList from './clue-list';
import * as Clue from './clue';

//
//
interface SentenceBase {
    num: number;
    text: string;
    combinations: string[];
}

export interface Variations {
    components: Record<string, string[]>;
    synonyms: Record<string, string[]>;
    homophones: Record<string, string[]>;
}

type NameCountMap = {
    [key: string]: number;
};

// run-time only, not part of schema
export interface Candidate {
    names: NameCountMap;
    clues: ClueList.Primary;
}

interface CandidatesContainer {
    candidates: Candidate[];
}

export type Type = SentenceBase & Variations & CandidatesContainer;
export type List = Type[];

export let load = (dir: string, num: number): Type => {
    return makeFrom(getFilename(dir, num));
};

export const emptyVariations = (): Variations => {
    return {
	components: {},
	synonyms: {},
	homophones: {}
    }
}

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
    return validateCombinations(sentence) && validateVariations(sentence);
}

const validateCombinations = (sentence: Type): boolean => {
    const sortedText = stripAndSort(sentence.text);
    for (let combo of sentence.combinations) {
	if (sortedText != stripAndSort(combo)) {
	    console.error(`combination: ${sortedText} != ${stripAndSort(combo)}`);
	    return false;
	}
    }
    return true;
}

const validateVariations = (sentence: Type): boolean => {
    for (let key of Object.keys(sentence.components)) {
	const sortedText = stripAndSort(key);
	for (let component of sentence.components[key]) {
	    if (sortedText != stripAndSort(component)) {
		console.error(`variation: ${sortedText} != ${stripAndSort(component)}`);
		return false;
	    }
	}
    }
    return true;
}

export const addVariations = (sentence: Type, variations: Variations): void => {
    for (let key of Object.keys(sentence.components)) {
	if (_.has(variations.components, key)) {
	    // this could be more relaxed; i could compare values to ensure equality
	    throw new Error(`duplicate component(${sentence.num}): ${key}`)
	}
	variations.components[key] = sentence.components[key];
	for (let component of variations.components[key]) {
	    if (component == key) continue;
	    if (_.has(variations.components, component)) {
		// this could be more relaxed; i could compare values to ensure equality
		throw new Error(`duplicate component(${sentence.num}) in ${key}: ${component}`)
	    }
	    variations.components[component] = variations.components[key];
	}
    }
};

export const buildAllCandidates = (sentence: Type, variations: Variations): Candidate[] => {
    let candidates: Candidate[] = [];
    let src = 1000000 * sentence.num; // up to 10000 variations of up to 100 names
    const sortedText = stripAndSort(sentence.text);
    for (const combo of sentence.combinations) {
	//console.error(`combo: ${combo}`);
	const nameListMap = buildCandidateNameLists(combo.split(' '), variations);
	for (let nameList of nameListMap.values()) {
	    //console.error(`nameList: ${nameList}`);
	    if (sortedText != joinAndSort(nameList)) {
		throw new Error(`sentence '${sentence.text}' != nameList '${nameList}'`);
	    }
	    candidates.push({
		names: buildNameCountMap(nameList),
		clues: buildClueList(sentence.num, nameList, src, variations)
	    });
	    src += 100;
	}
    }
    return candidates;
};

const buildNameCountMap = (nameList: string[]): NameCountMap => {
    let map: NameCountMap = {};
    for (let name of nameList) {
	if (!_.has(map, name)) {
	    map[name] = 1;
	} else {
	    map[name] += 1;
	}
    }
    return map;
};

const buildClueList = (num: number, nameList: string[], src: number,
    variations: Variations): ClueList.Primary =>
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

// TODO: alien technology. revisit or ignore at your peril.
// pretty sure this has some profound wrongness about it.
//
const buildCandidateNameLists = (components: string[], variations: Variations, startIndex = 0,
    results = new Map<string, string[]>()): Map<string, string[]> => 
{
    const log = false;
    if (log) {
	console.error(`  IN: ${components} @ ${components[startIndex]} (${startIndex} of ${components.length})`);
    }
    for (let i = startIndex; i < components.length; ++i) {
	const component = components[i];
	const replace = _.has(variations.components, component);
	const alternates: string[] = replace ? variations.components[component] : [component];
	for (let alternate of alternates) {
	    let copy = components.slice(0, i);
	    let offset = 0;
	    const split = alternate.split(' ');
	    if (split.length > 1) {
		copy.push(...split);
	    } else {
		copy.push(alternate);
		offset = 1;
	    }
	    for (let j = i + 1; j < components.length; ++j) {
		copy.push(components[j]);
	    }
	    const nextIndex = i + offset;
	    if (/*(startIndex < components.length - 1) || */(nextIndex < copy.length)) {
		buildCandidateNameLists(copy, variations, nextIndex, results);
	    } else if ((startIndex == components.length - 1) && (nextIndex === copy.length)) {
		const key = copy.slice().sort().join('');
		if (!results.has(key)) {
		    results.set(key, copy);
		    if (log) {
			console.error(`  OUT: ${copy} @ start(${components[startIndex]}), next(${copy[nextIndex]})` +
			    `, startIndex ${startIndex} of ${components.length}, nextIndex ${nextIndex} of ${copy.length}`);
		    }
		}
	    } else {
		if (log) {
		    console.error(`  skip: ${copy} @ start(${components[startIndex]}), next(${copy[nextIndex]})` +
			`, startIndex ${startIndex} of ${components.length}, nextIndex ${nextIndex} of ${copy.length}`);
		}
	    }
	}
    }
    return results;
};

// "strip" spaces from string, sort resulting letters
const stripAndSort = (text: string): string => {
    return joinAndSort(text.split(' '));
};

// join strings, sort resulting letters
const joinAndSort = (arr: string[]): string => {
    return arr.join('').split('').sort().join('');
}
