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

//
// args:
//  count:   # of primary clues to combine
//  max:     max # of sources to use
//  use:     list of clue names and name:counts, also allowing pairs, e.g. ['john:1','bob','red,bird']
//  // not supported: require: required clue counts, e.g. [3,5,8]
//  // not supported: limit to these primary sources, e.g. [1,9,14]
//
// A "clueSourceList" is a list (array) where each element is a
// object that contains a list (cluelist) and a count, such as
// [ { list:clues1, count:1 },{ list:clues2, count:2 }].
//
let makeCombos = function(args, options = {}) {
    this.nextDupeClue = 0;
    this.nextDupeSrc = 0;
    this.nextDupeCombo = 0;

    // TODO USE "validateArgs" 

    /*
    let require = args.require ? _.clone(args.require) : [];
    if (!_.isEmpty(args.require)) throw new Error('require not yet supported');
    if (args.sources) throw new Error('sources not yet supported');
    */

    //this.hash = {};
    let allCombos = [];

    // TODO: push out parallelization a level, compute these once per process

    let allXorNcDataLists = args.xor ? buildAllUseNcDataLists(args.xor) : [ [] ];
    let allAndNcDataLists = args.and ? buildAllUseNcDataLists(args.and) : [ [] ];
    let allOrNcDataLists = args.or ? buildAllUseNcDataLists(args.or) : [ [] ];

    //let allXorNcLists = args.xor ? buildAllUseNcLists(args.xor) : [ [] ];
    //console.log(`allXorNcDataLists: ${Stringify(allXorNcDataLists)}`);
    //console.log(`allXorNcDataLists[0]: ${Stringify(allXorNcDataLists[0])}`);

    let comboArgs = {
        sum: args.sum,
        max: args.max,
	allXorNcDataLists,
	allAndNcDataLists,
	allOrNcDataLists
    };
    
    let combos = getCombosForUseNcLists(comboArgs, options);
    allCombos.push(...combos);

    //Debug(`dupeClue(${this.nextDupeClue})` +
    //`, dupeSrc(${this.nextDupeSrc})` +
    //`, dupeCombo(${this.nextDupeCombo})`);

//    console.error(`timing: ${PrettyMs(timing)} mcsl: ${mcsl_timing/MILLY}ms sl: ${sl_timing/MILLY}ms maus: ${maus_timing/MILLY}ms`);
//    console.error(`mcsl iter: ${mcsl_iter} ms iter: ${ms_iter} avg ms/mscl: ${_.toInteger(ms_iter / mcsl_iter)}`);
//    console.error(`usenc count: ${usenc_count} size: ${usenc_size} avg: ${_.toInteger(usenc_size/usenc_count)}`);
//    console.error(`usenc sources count: ${usenc_sources_count} size: ${usenc_sources_size} avg: ${_.toInteger(usenc_sources_size/usenc_sources_count)}`);
//    console.error(`sources count: ${sources_count} size: ${sources_size} avg: ${_.toInteger(sources_size/sources_count)}`);

    return allCombos;
};


        const comboList = makeCombos(args);
        total += comboList.length;
        const filterResult = ClueManager.filter(comboList, sum, comboMap);
        known += filterResult.known;
        reject += filterResult.reject;
        duplicate += filterResult.duplicate;
    }

module.exports = {
    doWork
};
