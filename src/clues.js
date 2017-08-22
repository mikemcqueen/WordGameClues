//
// clues.js
//

'use strict';

const _           = require('lodash');
const AltSources  = require('./alt-sources');
const ClueList    = require('./clue-list');
const ClueManager = require('./clue-manager');
const Clues       = require('./clue-types');
const ComboMaker  = require('./combo-maker');
const ComboSearch = require('./combo-search');
const Components  = require('./show-components');
const Duration    = require('duration');
const Expect      = require('chai').expect;
const NameCount   = require('./name-count');
const Peco        = require('./peco');
const PrettyMs    = require('pretty-ms');
const Show        = require('./show');
const StringifyObject = require('stringify-object');
const ResultMap   = require('./result-map');
const Validator   = require('./validator');

// initialize command line options.  do this before logger.
//

// metamorphois -> synthesis -> harmonize -> finalize

// TODO:
// solve the -p/-y problem: standardize how to set/initialize state based on target clues
// 

const Opt = require('node-getopt')
    .create(_.concat(Clues.Options, [
	['a', 'alt-sources=NAME',                    'show alternate sources for the specified clue'],
	['A', 'all-alt-sources',                     'show alternate sources for all clues'],
	['o', 'output',                              '  output json -or- clues(huh?)'],
	['c', 'count=COUNT[LO,COUNTHI]',             '# of primary clues to combine; or range if COUNTHI is specified'],
	['d', 'allow-dupe-source',                   'allow duplicate source, override default behavior of --meta'],
	['i', 'primary-sources=SOURCE[,SOURCE,...]', 'limit results to the specified primary source(s)'],
	['',  'inverse',                             '  or the inverse of those source(s); use with -i'],
	['k', 'show-known',                          'show compatible known clues; -u <clue> required' ],
	['',  'csv',                                 '  output in search-term csv format' ],
	['',  'files',                               '  output in result file full-path format' ],
	['q', 'require-counts=COUNT+',               'require clue(s) of specified count(s)' ],
	['s', 'show-sources=NAME[:COUNT][,v]',       'show primary source combinations for the specified name[:count]' ],
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

    const options = Opt.options;

    // TODO: get rid of this, just pass Opt.options around
    let countArg = options.count;
    let maxArg = _.toNumber(options.max);
    let useClueList = options.use;
    let showSourcesClueName = options['show-sources'];
    let altSourcesArg = options['alt-sources'];
    let allAltSourcesFlag = options['all-alt-sources'];
    let allowDupeSrcFlag = options['allow-dupe-source'];
    let showKnownArg = options['show-known'];

    if (!maxArg) {
	maxArg = 2; // TODO: default values in opt
    }
    if (_.isUndefined(options.count)) {
	needCount = true;

	if (showSourcesClueName ||
	    altSourcesArg ||
	    allAltSourcesFlag ||
	    showKnownArg ||
	    options.test ||
	    useClueList
	   ) {
	    needCount = false;
	}
	if (needCount) {
	    console.log('-c COUNT required with those options');
	    return 1;
	}
    }

    if (options.output) {
	QUIET = true;
    }

    Validator.setAllowDupeFlags({
	allowDupeNameSrc: false,
	allowDupeSrc:     (allowDupeSrcFlag ? true : false),
	allowDupeName:    true
    });

    if (_.includes(options.flags, '1')) {
	validateAllOnLoad = true;
	console.log('validateAllOnLoad=true');
    }
    if (_.includes(options.flags, '2')) {
	ignoreLoadErrors = true;
	console.log('ignoreLoadErrors=true');
    }

    let clueSource = Clues.getByOptions(options);

    setLogging(_.includes(options.verbose, VERBOSE_FLAG_LOAD));
    if (!loadClues(clueSource, validateAllOnLoad, ignoreLoadErrors)) {
	return 1;
    }
    setLogging(options.verbose);

    log(`count=${options.count}, max=${maxArg}`);

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
	    max:      options.count ? _.toNumber(options.count) : ClueManager.maxClues,
	    root:     '../data/results/',
	    format:   {
		csv:   options.csv,
		files: options.files
	    }
	});
    }
    else if (options.test) {
	Components.show(options);
    }
    else if (showSourcesClueName) {
	showSources(showSourcesClueName);
    }
    else if (altSourcesArg || allAltSourcesFlag) {
	AltSources.show(allAltSourcesFlag ? {
	    all    : true,
	    output : options.output,
	    count  : _.toNumber(options.count)
	} : {
	    all    : false,
	    name   : altSourcesArg,
	    output : options.output
	});
    }
    else {
	let sources = options['primary-sources'];
	if (options.inverse) {
	    sources = ClueManager.getInversePrimarySources(sources.split(',')).join(',');
	    console.log(`inverse sources: ${sources}`);
	}
	doCombos({
	    sum:     options.count,
	    max:     maxArg,
	    require: options['require-counts'],
	    sources,
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
//	    result.resultMap.dump();
	    console.log(StringifyObject(result.resultMap.map(), { indent: '  ' }));
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
    AltSources.logging  = flag;
    ComboSearch.logging = flag;
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
