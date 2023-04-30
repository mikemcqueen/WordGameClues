//
// source.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const ResultMap   = require('../../types/result-map');
const Peco        = require('../../modules/peco');

const Assert      = require('assert');
const Debug       = require('debug')('source');
const Duration    = require('duration');
const PrettyMs    = require('pretty-ms');
//const stringify   = require('javascript-stringify').stringify;
const Stringify  = require('stringify-object');

//import * as Clue from '../types/clue';
//import * as ClueManager from './clue-manager';
import * as CountBits from '../types/count-bits-fastbitset';
import * as NameCount from '../types/name-count';
import * as Sentence from '../types/sentence';

import { ValidateResult } from './validator';

interface Base {
    primaryNameSrcList: NameCount.List;
    ncList: NameCount.List;
}

export interface CompatibilityData {
    sourceBits: CountBits.Type;
    usedSources: number[];
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

///////

export const makeData = (nc: NameCount.Type, validateResult: ValidateResult):
    Data =>
{
    Assert(validateResult.sourceBits && validateResult.usedSources, `makeData(): ${nc}`);
    return {
	primaryNameSrcList: validateResult.nameSrcList,
	sourceBits: validateResult.sourceBits,
	//usedSources: validateResult.usedSources, //TODO?
	usedSources: Sentence.getUsedSources(validateResult.nameSrcList),
        ncList: [nc],
    };
};

export const isXorCompatible = (first: CompatibilityData,
    second: CompatibilityData): boolean =>
{
    if (CountBits.intersects(first.sourceBits, second.sourceBits)) {
	return false;
    }
    for (let i = 1; i < 10 /* cough */; ++i) {
	if ((first.usedSources[i] && second.usedSources[i]) &&
	    (first.usedSources[i] !== second.usedSources[i]))
	{
	    return false;
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

