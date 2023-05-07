//
// source.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Peco        = require('../../modules/peco');
const ResultMap   = require('../../types/result-map');

const Assert      = require('assert');
const Debug       = require('debug')('source');
const Duration    = require('duration');
const PrettyMs    = require('pretty-ms');
const Stringify  = require('stringify-object');

import * as CountBits from '../types/count-bits-fastbitset';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';

import { ValidateResult } from './validator';

//////////

interface Base {
    primaryNameSrcList: NameCount.List;
    ncList: NameCount.List;
}

type UsedSourceSet = Set<number>;
export type UsedSources = UsedSourceSet[];

export interface CompatibilityData {
    sourceBits: CountBits.Type;
    usedSources: UsedSources;
}
export type Data = Base & CompatibilityData;
export type List = Data[];

interface ValidateResultsContainer {
    validateResults: ValidateResult[];
}
export type LazyData = Base & CompatibilityData & ValidateResultsContainer;
export type AnyData = LazyData | Data;

export interface ListContainer {
    sourceList: List;
}

//////////

export const isCandidate = (src: number): boolean => {
    return src >= 1_000_000;
};

export const getCandidateSentence = (src: number): number => {
    Assert(isCandidate(src));
    return Math.floor(src / 1_000_000);
};

const getCandidateSource = (src: number): number => {
    Assert(isCandidate(src));
    return src % 1_000_000;
}

const getVariation = (src: number): number => {
    // no isCandidate check, since this is (could be) already the % 1_000_00 value
    return Math.floor((src % 1_000_000) / 100);
}

export const isXorCompatible = (first: CompatibilityData,
    second: CompatibilityData): boolean =>
{
    if (CountBits.intersects(first.sourceBits, second.sourceBits)) {
	return false;
    }
    for (let i = 1; i < 10 /* cough */; ++i) {
	if ((first.usedSources[i] === undefined) ||
	    (second.usedSources[i] === undefined))
	{
	    continue; // one or both undefined, is compatible
	}
	const [firstElem] = first.usedSources[i];
	const [secondElem] = second.usedSources[i];
	if (getVariation(firstElem) !== getVariation(secondElem)){
	    return false; // variation incompatibility
	}
	for (let firstSrc of first.usedSources[i]) {
	    if (second.usedSources[i].has(firstSrc)) {
		return false; // index incompatibility
	    }
	}
    }
    return true;
}

export const isXorCompatibleWithAnySource = (source: CompatibilityData,
    sourceList: CompatibilityData[]): boolean =>
{
    let compatible = sourceList.length === 0; //listIsEmpty(sourceList); // empty list == compatible
    for (let otherSource of sourceList) {
        compatible = isXorCompatible(source, otherSource);
        if (compatible) break;
    }
    return compatible;
};

//////////

// return false if source is incompatible, true otherwise
export const addUsedSource = (usedSources: UsedSources, src: number, nothrow = false):
    boolean =>
{
    if (!isCandidate(src)) return true;
    const sentence = getCandidateSentence(src);
    if (usedSources[sentence] === undefined) {
	usedSources[sentence] = new Set<number>();
    }
    let set = usedSources[sentence];
    const source = getCandidateSource(src);
    // defensive incompatible variation index check
    if (set.size) {
	// trick to get first elem from set.
	const [anyElem] = set;
	if (getVariation(anyElem) !== getVariation(source)) {
	    Debug(`oopsie ${anyElem} (${getVariation(anyElem)})` +
		`, ${source} (${getVariation(source)})`);
	    if (nothrow) return false;
	    throw new Error(`oopsie`);
	}
	if (set.has(source)) {
	    Debug(`poopsie ${source}, [${[...set]}]`);
	    if (nothrow) return false;
	    throw new Error(`poopsie`);
	}
    }
    set.add(source);
    return true;
}

export const getUsedSources = (nameSrcList: NameCount.List):
    UsedSources =>
{
    let result: UsedSources = [];
    nameSrcList.filter(nameSrc => isCandidate(nameSrc.count))
	.forEach(nameSrc => addUsedSource(result, nameSrc.count));
    return result;
}

const addAll = (set: Set<number>, values: number[]): void => {
    for (let value of values) {
        set.add(value);
    }
}

export const mergeUsedSourcesInPlace = (to: UsedSources, from: UsedSources):
    void =>
{
    for (let i = 1; i < 10 /* cough */; ++i) {
	const un_to = to[i] === undefined;
        const un_from = from[i] === undefined;
        if (un_to && un_from) continue;
        if (un_to) {
            to[i] = new Set<number>();
        }
        const toSize = to[i].size;
        if (!un_from) {
            addAll(to[i], [...from[i]]);
        }
        Assert(to[i].size === (toSize + (from[i]?.size || 0)));
    }
}

export const mergeUsedSources = (first: UsedSources, second: UsedSources):
    UsedSources =>
{
    let result: UsedSources = [];
    for (let i = 1; i < 10 /* cough */; ++i) {
	if ((first[i] === undefined) && (second[i] === undefined)) {
	    continue;
	}
	const firstValues = (first[i] !== undefined) ? [...first[i]] : [];
	const secondValues = (second[i] !== undefined) ? [...second[i]] : [];
	//result[i] = new Set([...firstValues, ...secondValues]);
	let set = new Set<number>();
        addAll(set, firstValues)
        addAll(set, secondValues);
        result[i] = set;
        if (result[i].size !== ((first[i]?.size || 0) + (second[i]?.size || 0))) {
            console.error(`result[${i}]: ${result[i].size} != ${first[i]?.size || 0} + ${second[i]?.size || 0}`);
            console.error(`  first: [${firstValues}], second: [${secondValues}]`);
            Assert(false);
        }
    }
    return result;
}

export const makeData = (nc: NameCount.Type, validateResult: ValidateResult):
    Data =>
{
    Assert(validateResult.sourceBits && validateResult.usedSources, `makeData(): ${NameCount.toString(nc)}`);
    Assert(NameCount.listHasCompatibleSources(validateResult.nameSrcList), `makeData(): ${NameCount.toString(nc)}`);
    return {
	primaryNameSrcList: validateResult.nameSrcList,
	sourceBits: validateResult.sourceBits,
	//usedSources: validateResult.usedSources, //TODO?
	usedSources: getUsedSources(validateResult.nameSrcList),
        ncList: [nc],
    };
};
