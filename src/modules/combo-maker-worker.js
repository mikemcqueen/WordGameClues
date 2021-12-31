'use strict';

const ClueManager = require('./clue-manager');
const ComboMaker  = require('./combo-maker');
const NameCount   = require('../types/name-count');

let doWork = (listOfNcListStr) => {
    let hash = {};
    let combos = [];

    for (const ncListStr of listOfNcListStr) {
	const ncList = NameCount.makeListFromCsv(ncListStr);

	let sources;
	if (!hash[ncListStr]) {
	    sources = ComboMaker.mergeAllCompatibleSources(ncList, 'c4UNCL');
	    hash[ncListStr] = { sources };
	} else {
	    //cacheHitCount += 1;
	}
	sources = hash[ncListStr].sources;
	
	let logging = 0;
	if (logging) console.log(`  found compatible sources: ${!_.isEmpty(sources)}`);

        // failed to find any compatible combos
        if (_.isEmpty(sources)) continue;

	let nameList;
	if (!hash[ncListStr].isUseNcCompatible) {
	    nameList = NameCount.makeNameList(ncList);
	    hash[ncListStr].isUseNcCompatible = ComboMaker.isCompatibleWithUseNcLists(sources, args, nameList, options);
	}
	if (hash[ncListStr].iUseNcCompatible) {
	    //compatibleVariations += 1;
            combos.push(nameList.toString());
        }
    }
    //totalVariationCount += variationCount;

    /*
    console.error(`combos(${comboCount}), variations(${totalVariationCount}), compatible(${combos.length}), ` +
		  `AVG variations/combo(${totalVariationCount/comboCount}), ` +
		  `cacheHits(${cacheHitCount}), ${PrettyMs(duration)}`);
    */
    
    return combos;
};

module.exports = {
    doWork
};
