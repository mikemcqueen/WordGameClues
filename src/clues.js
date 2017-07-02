//
// CLUES.JS
//

'use strict';

let _           = require('lodash');
let AltSources  = require('./alt_sources');
let ClueList    = require('./clue_list');
let ClueManager = require('./clue_manager');
let Clues       = require('./clue-types');
let ComboMaker  = require('./combo_maker');
let ComboSearch = require('./combo_search');
let Components  = require('./show-components');
let Duration    = require('duration');
let Expect      = require('chai').expect;
let NameCount   = require('./name_count');
let Peco        = require('./peco');
let PrettyMs    = require('pretty-ms');
let Show        = require('./show');
let ResultMap   = require('./resultmap');
let Validator   = require('./validator');

// initialize command line options.  do this before logger.
//

// metamorphois -> synthesis -> harmonize -> finalize

// TODO:
// solve the -p/-y problem: standardize how to set/initialize state based on target clues
// 

let Opt = require('node-getopt')
    .create(_.concat(Clues.Options, [
	['a', 'alt-sources=NAME',            'show alternate sources for the specified clue' ],
	['A', 'all-alt-sources',             'show alternate sources for all clues' ],
	['o', 'output',                      '  output json -or- clues(huh?)' ],
	['c', 'count=COUNT[LO,COUNTHI]',     '# of primary clues to combine; or range if COUNTHI is specified' ],
	['d', 'allow-dupe-source',           'allow duplicate source, override default behavior of --meta' ],
	['i', 'primary-sources=SOURCE[,SOURCE,...]', 'limit results to the specified primary source(s)' ],
	['k', 'show-known',                  'show compatible known clues; -u <clue> required' ],
	['',  'csv',                         '  output in search-term csv format' ],
	['q', 'require-counts=COUNT+',       'require clue(s) of specified count(s)' ],
	['s', 'show-sources=NAME[:COUNT][,v]', 'show primary source combinations for the specified name[:count]' ],
	['t', 'test=SOURCE[,SOURCE,...]',    'test the specified source list, e.g. blue,fish' ],
	['',  'add=NAME',                    '  add combination to known list as NAME; use with --test' ],
	['',  'reject',                      '  add combination to reject list; use with --test' ],
	['u', 'use=NAME[:COUNT]+',           'use the specified name[:count](s)' ],
	['x', 'max=COUNT',                   'specify maximum # of components to combine'],
	['z', 'flags=OPTION+',               'flags: 1=validateAllOnLoad,2=ignoreLoadErrors' ],

	['v', 'verbose=OPTION+',             'show logging. OPTION=load' ],
	['h', 'help',                        'this screen']
    ]))
    .bindHelp().parseSystem();

//

let LOGGING = false;
let QUIET = false;

//

const VERBOSE_FLAG_LOAD     = 'load';

//
//

function main () {
    let needCount;
    let validateAllOnLoad;
    let ignoreLoadErrors;

    // options

    // TODO: get rid of this, just pass Opt.options around
    let countArg = Opt.options.count;
    let maxArg = _.toNumber(Opt.options.max);
    let useClueList = Opt.options.use;
    let verboseArg = Opt.options['verbose'];
    let showSourcesClueName = Opt.options['show-sources'];
    let altSourcesArg = Opt.options['alt-sources'];
    let allAltSourcesFlag = Opt.options['all-alt-sources'];
    let allowDupeSrcFlag = Opt.options['allow-dupe-source'];
    let showKnownArg = Opt.options['show-known'];
    let flagsArg = Opt.options.flags;

    if (!maxArg) {
	maxArg = 2; // TODO: default values in opt
    }
    if (_.isUndefined(Opt.options.count)) {
	needCount = true;

	if (showSourcesClueName ||
	    altSourcesArg ||
	    allAltSourcesFlag ||
	    showKnownArg ||
	    Opt.options.test ||
	    useClueList
	   ) {
	    needCount = false;
	}
	if (needCount) {
	    console.log('-c COUNT required with those options');
	    return 1;
	}
    }

    if (Opt.options.output) {
	QUIET = true;
    }

    Validator.setAllowDupeFlags({
	allowDupeNameSrc: false,
	allowDupeSrc:     (allowDupeSrcFlag ? true : false),
	allowDupeName:    true
    });

    if (_.includes(flagsArg, '1')) {
	validateAllOnLoad = true;
	console.log('validateAllOnLoad=true');
    }
    if (_.includes(flagsArg, '2')) {
	ignoreLoadErrors = true;
	console.log('ignoreLoadErrors=true');
    }

    let clueSource = Clues.getByOptions(Opt.options);

    setLogging(_.includes(verboseArg, VERBOSE_FLAG_LOAD));
    if (!loadClues(clueSource, validateAllOnLoad, ignoreLoadErrors)) {
	return 1;
    }
    setLogging(verboseArg);

    log(`count=${Opt.options.count}, max=${maxArg}`);

    // TODO: add "show.js" with these exports
    if (showKnownArg) {
	if (!useClueList) {
	    // TODO: require max if !metaFlag
	    console.log('one or more -u NAME:COUNT required with that option');
	    return 1;
	}
	/*
	if (!metaFlag) {
	    console.log('--meta required with that option');
	    return 1;
	}
	 */
	Show.compatibleKnownClues({
	    nameList: useClueList,
	    max:      _.isUndefined(Opt.options.count)
		? ClueManager.maxClues
		: _.toNumber(Opt.options.count),
	    asCsv:    Opt.options.csv
	});
    }
    else if (Opt.options.test) {
	Components.show(Opt.options);
    }
    else if (showSourcesClueName) {
	showSources(showSourcesClueName);
    }
    else if (altSourcesArg || allAltSourcesFlag) {
	AltSources.show(allAltSourcesFlag ? {
	    all    : true,
	    output : Opt.options.output,
	    count  : _.toNumber(Opt.options.count)
	} : {
	    all    : false,
	    name   : altSourcesArg,
	    output : Opt.options.output
	});
    }
    else {
	doCombos({
	    sum:     Opt.options.count,
	    max:     maxArg,
	    require: Opt.options['require-counts'],
	    sources: Opt.options['primary-sources'],
	    use:     useClueList
	});
    }
}

//

function loadClues (clues, validateAllOnLoad, ignoreLoadErrors) {
    log('loading all clues...');
    ClueManager.loadAllClues({
	clues,
	validateAll:  validateAllOnLoad,
	ignoreErrors: ignoreLoadErrors
    });
    log('done.');
    return true;
}

//
// args:
//  sum:     countArg,
//  max:     maxArg,
//  require: requiredSizes,
//  sources: primarySourcesArg,
//  use:     useClueList
//

function doCombos(args) {
    if (!_.isUndefined(args.sources)) {
	args.sources = _.chain(args.sources).split(',').map(_.toNumber).value();
    }
    if (!_.isUndefined(args.require)) {
	args.require = _.chain(args.require).split(',').map(_.toNumber).value();
    }
    let sumRange;
    if (!_.isUndefined(args.sum)) {
	sumRange = _.chain(args.sum).split(',').map(_.toNumber).value();
    }
    Expect(sumRange, 'invalid sumRange').to.be.an('array').with.length.of.at.most(2);
    console.log('++combos' +
		`, sum: ${sumRange}` +
		`, max: ${args.max}` +
		`, require: ${args.require}` +
		`, sources: ${args.sources}` +
		`, use: ${args.use}`);

    let total = 0;
    let known = 0;
    let reject = 0;
    let duplicate  = 0;
    let comboMap = {};
    let beginDate = new Date();
    let lastSum = sumRange.length > 1 ? sumRange[1] : sumRange[0];
    for (let sum = sumRange[0]; sum <= lastSum; ++sum) {
	args.sum = sum;
	let max = args.max;
	if (args.max > args.sum) args.max = args.sum;
	const comboList = ComboMaker.makeCombos(args);
	args.max = max;
	total += comboList.length;
	const filterResult = ClueManager.filter(comboList, args.sum, comboMap);
	known += filterResult.known;
	reject += filterResult.reject;
	duplicate += filterResult.duplicate;
    }
    let d = new Duration(beginDate, new Date()).milliseconds;
    _.keys(comboMap).forEach(nameCsv => console.log(nameCsv));
    console.log(`total: ${total}` +
		', filtered: ' + _.size(comboMap) +
		', known: ' + known +
		', reject: ' + reject +
		', duplicate: ' + duplicate);
    console.log(`--combos: ${PrettyMs(d)}`);

    if (total !== _.size(comboMap) + known + reject + duplicate) {
	console.log('WARNING: amounts to not add up!');
    }
}

//
//

function showSources(clueName) {
    let result;
    let nc;
    let verbose;
    let clueSplitList = clueName.split(',');

    clueName = clueSplitList[0];
    nc = NameCount.makeNew(clueName);
    if (_.size(clueSplitList) > 1) {
	verbose = clueSplitList[1] === 'v';
    }
    if (!nc.count) {
	throw new Error('Need to supply a count as name:count (for now)');
    }

    log('++sources');

    result = Validator.validateSources({
	sum:          nc.count,
	nameList:     [ nc.name ],
	count:        1,
	validateAll:  true
    });
    if (result.success) {
	result.list.forEach(result => {
	    console.log('nameSrcList: ' + result.nameSrcList);
	    if (verbose) {
		console.log('ncList:      ' + result.ncList);
		result.resultMap.dump();
	    }
	});
    }
    else {
	console.log('validate failed.');
    }
}

//

function setLogging(flag) {
    ClueManager.logging = flag;
    ComboMaker.logging  = flag;
    Validator.setLogging(flag);
    AltSources.logging  = flag;
    ComboSearch.logging = flag;
    ResultMap.setLogging(flag);
    //Peco.setLogging(flag);

    LOGGING = flag;
}

//

function log(text) {
    if (LOGGING) {
	console.log(text);
    }
}

//
//

let appBegin;


try {
    appBegin = new Date();
    main();
}
catch(e) {
    console.error(e.stack);
}
finally {
    if (LOGGING && !QUIET) {
	console.log('runtime: ' + (new Duration(appBegin, new Date())).seconds + ' seconds');
    }
}
