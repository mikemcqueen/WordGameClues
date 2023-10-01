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

const kNumSentences = 9;
const kMaxSourcesPerSentence = 32;

//////////

interface Base {
    primaryNameSrcList: NameCount.List;
    ncList: NameCount.List;
}

type Variations = Int16Array;
export interface UsedSources {
    bits: CountBits.Type;
    variations: Variations;
}

export interface CompatibilityData {
    usedSources: UsedSources;
}
export type Data = Base & CompatibilityData;
export type List = Data[];

/*
interface ValidateResultsContainer {
    validateResults: ValidateResult[];
}
export type LazyData = Base & CompatibilityData & ValidateResultsContainer;
export type AnyData = LazyData | Data;
*/

export interface ListContainer {
    sourceList: List;
}

//////////

export const emptyUsedSources = (): UsedSources => {
    let usedSources: UsedSources = {
        bits: CountBits.makeNew(),
        variations: new Int16Array(kNumSentences)
    };
    for (let i = 0; i < kNumSentences; ++i) {
        usedSources.variations[i] = -1;
    }
    return usedSources;
}

export const isCandidate = (src: number): boolean => {
    return src >= 1_000_000;
};

export const getCandidateSentence = (src: number): number => {
    Assert(isCandidate(src));
    return Math.floor(src / 1_000_000);
};

/*
const getCandidateSource = (src: number): number => {
    Assert(isCandidate(src));
    return src % 1_000_000;
}
*/

const getCandidateVariation = (src: number): number => {
    // no isCandidate check, since this is (could be) already the % 1_000_000 value
    return Math.floor((src % 1_000_000) / 100);
}

const getIndex = (src: number): number => {
    // no isCandidate check, since this is (could be) already the % 1_000_000 value
    return Math.floor((src % 1_000_000) % 100);
}

const getFirstBitIndex = (sentence: number): number => {
    Assert(sentence > 0);
    return (sentence - 1) * kMaxSourcesPerSentence;
}

const allVariationsEqual = (v1: Variations, v2: Variations): boolean => {
    for (let i = 0; i < v1.length; ++i) {
        if (v1[i] !== v2[i]) {
            return false;
        }
    }
    return true;
}

export const isEqual = (first: CompatibilityData,
    second: CompatibilityData): boolean =>
{
    if (!CountBits.equals(first.usedSources.bits, second.usedSources.bits)) {
        return false;
    }
    if (!allVariationsEqual(first.usedSources.variations,
        second.usedSources.variations))
    {
        return false;
    }
    return true;
}

const allVariationsXorCompatible = (v1: Variations, v2: Variations): boolean => {
    for (let i = 0; i < v1.length; ++i) {
      if ((v1[i] > -1) && (v2[i] > -1) && (v1[i] !== v2[i])) {
        return false;
      }
    }
    return true;
}

export const isXorCompatible = (first: CompatibilityData,
    second: CompatibilityData, check_variations: boolean = true): boolean =>
{
    if (CountBits.intersects(first.usedSources.bits, second.usedSources.bits)) {
        return false;
    }
    if (check_variations && !allVariationsXorCompatible(
        first.usedSources.variations, second.usedSources.variations))
    {
        return false;
    }
    return true;
}

export const isXorCompatibleWithAnySource = (source: CompatibilityData,
    sourceList: CompatibilityData[]): boolean =>
{
    let compatible = sourceList.length === 0;
    for (let otherSource of sourceList) {
        compatible = isXorCompatible(source, otherSource);
        if (compatible) break;
    }
    return compatible;
};

//////////

const getVariation = (usedSources: UsedSources, sentence: number): number => {
    return usedSources.variations[sentence - 1];
}

const hasVariation = (usedSources: UsedSources, sentence: number): boolean => {
    return getVariation(usedSources, sentence) > -1;
}

const setVariation = (usedSources: UsedSources, sentence: number, variation: number): void => {
    usedSources.variations[sentence - 1] = variation;
}

// if source is incompatible, throw execption, or return false if nothrow is true.
// return true if compatible
export const addUsedSource = (usedSources: UsedSources, src: number, nothrow = false):
    boolean =>
{
    const sentence = getCandidateSentence(src);
    const variation = getCandidateVariation(src);
    if (hasVariation(usedSources, sentence) && (getVariation(usedSources, sentence) !== variation)) {
        if (nothrow) return false;
        console.error(`variation(${sentence}), this: ${getVariation(usedSources, sentence)}` +
            `, src: ${variation}`);
        Assert(!"variation mismatch");
    }
    const index = getIndex(src);
    Assert(index < kMaxSourcesPerSentence);
    
    setVariation(usedSources, sentence, variation);

    const bit_pos = index + getFirstBitIndex(sentence);
    if (CountBits.test(usedSources.bits, bit_pos)) {
        if (nothrow) return false;
        Assert(!"incompatible bits");
    }
    CountBits.set(usedSources.bits, bit_pos);
    return true;
}

export const getUsedSources = (nameSrcList: NameCount.List):
    UsedSources =>
{
    let result: UsedSources = emptyUsedSources();
    nameSrcList.forEach(nameSrc => addUsedSource(result, nameSrc.count));
    return result;
}

const addVariations = (to: UsedSources, from: UsedSources): void => {
    for (let sentence = 1; sentence <= kNumSentences; ++sentence) {
        if (!hasVariation(from, sentence)) continue;
        if (hasVariation(to, sentence)) {
            Assert(getVariation(to, sentence) == getVariation(from, sentence));
        } else {
            setVariation(to, sentence, getVariation(from, sentence));
        }
    }
}

export const mergeUsedSourcesInPlace = (to: UsedSources, from: UsedSources):
    void =>
{
    // merge bits
    CountBits.orInPlace(to.bits, from.bits);
    // merge variations
    addVariations(to, from);
}

export const mergeUsedSources = (first: UsedSources, second: UsedSources):
    UsedSources =>
{
    let result: UsedSources = emptyUsedSources();
    mergeUsedSourcesInPlace(result, first);
    mergeUsedSourcesInPlace(result, second);
    return result;
}

/*
export const makeData = (nc: NameCount.Type, validateResult: ValidateResult):
    Data =>
{
    Assert(validateResult.usedSources, `makeData(): ${NameCount.toString(nc)}`);
    Assert(NameCount.listHasCompatibleSources(validateResult.nameSrcList),
        `makeData(): ${NameCount.toString(nc)}`);
    return {
	primaryNameSrcList: validateResult.nameSrcList,
        ncList: [nc],
	usedSources: validateResult.usedSources
    };
};
*/
