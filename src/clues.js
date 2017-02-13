//
// CLUES.JS
//

'use strict';

var _           = require('lodash');
var Duration    = require('duration');
var Np          = require('named-parameters');

var ClueManager = require('./clue_manager');
var ComboMaker  = require('./combo_maker');
var AltSources  = require('./alt_sources');
var ComboSearch = require('./combo_search');
var Validator   = require('./validator');
var ClueList    = require('./clue_list');
var NameCount   = require('./name_count');
var Peco        = require('./peco');
var Show        = require('./show');
var ResultMap   = require('./resultmap');

// initialize command line options.  do this before logger.
//

var Opt = require('node-getopt')
    .create([
	['a', 'alt-sources=NAME',            'show alternate sources for the specified clue' ],
	['A', 'all-alt-sources',             'show alternate sources for all clues' ],
	['c', 'count=COUNT',                 '# of primary clues to combine' ],
	['d', 'allow-dupe-source'  ,         'allow duplicate source, override default behavior of --meta' ],
	['' , 'json=WHICH',                  'specify which clue files to use. WHICH=meta|synth' ],
	['k', 'show-known'         ,         'show compatible known clues; -u <clue> required' ],
	['m', 'meta'               ,         'use metamorphosis clues, same as --json meta (default)' ],
	['o', 'output'             ,         'output json -or- clues' ],
	['p', 'primary-sources=SOURCE[,SOURCE,...]', 'limit results to the specified primary source(s)' ],
	['q', 'require-counts=COUNT+',       'require clue(s) of specified count(s)' ],
	['s', 'show-sources=NAME[:COUNT][,v]', 'show primary source combinations for the specified name[:count]' ],
	['t', 'test=SOURCE[,SOURCE,...]',    'test the specified source list, e.g. blue,fish' ],
	['',  'add=NAME',                    '  add combination to known list as NAME; use with --try' ],
	['',  'reject',                      '  add combination to reject list; use with --try' ],
	['u', 'use=NAME[:COUNT]+',           'use the specified name[:count](s)' ],
	['x', 'max=COUNT',                   'specify maximum # of components to combine'],
	['y', 'synthesis',                   'use synthesis clues, same as --json synth' ],
	['z', 'flags=OPTION+',               'flags: 1=validateAllOnLoad,2=ignoreLoadErrors' ],

	['v' , 'verbose=OPTION+',             'show logging. OPTION=load' ],
	['h' , 'help'               ,         'this screen']
    ])
    .bindHelp().parseSystem();

//

var LOGGING = false;
var QUIET = false;

//

const VERBOSE_FLAG_LOAD     = 'load';

//
//

function main() {
    var needCount;
    var validateAllOnLoad;
    var ignoreLoadErrors;

    // options

    // TODO: get rid of this, just pass Opt.options around
    let countArg = _.toNumber(Opt.options['count']);
    let maxArg = _.toNumber(Opt.options['max']);
    let requiredSizes = Opt.options['require-counts'];
    let useClueList = Opt.options['use'];
    let metaFlag = Opt.options['meta'];
    let verboseArg = Opt.options['verbose'];
    let outputArg = Opt.options['output'];
    let showSourcesClueName = Opt.options['show-sources'];
    let altSourcesArg = Opt.options['alt-sources'];
    let allAltSourcesFlag = Opt.options['all-alt-sources'];
    let allowDupeSrcFlag = Opt.options['allow-dupe-source'];
    let showKnownArg = Opt.options['show-known'];
    let synthFlag = Opt.options['synthesis'];
    let jsonArg = Opt.options['json'];
    let primarySourcesArg = Opt.options['primary-sources'];
    let flagsArg = Opt.options.flags;

    if (!maxArg) {
	maxArg = 2; // TODO: default values in opt
    }
    if (!countArg) {
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

    if (outputArg) {
	QUIET = true;
    }

    Validator.setAllowDupeFlags(synthFlag ? {
	allowDupeNameSrc: false,
	allowDupeSrc:     true,
	allowDupeName:    true,
    } : {
	allowDupeNameSrc: false,
	allowDupeSrc:     (allowDupeSrcFlag ? true : false),
	allowDupeName:    true,
    });

    if (_.includes(flagsArg, '1')) {
	validateAllOnLoad = true;
	console.log('validateAllOnLoad=true');
    }
    if (_.includes(flagsArg, '2')) {
	ignoreLoadErrors = true;
	console.log('ignoreLoadErrors=true');
    }

    setLogging(_.includes(verboseArg, VERBOSE_FLAG_LOAD));

    //
    if (!loadClues(synthFlag, metaFlag, jsonArg, validateAllOnLoad, ignoreLoadErrors)) {
	return 1;
    }

    setLogging(verboseArg);

    // hacky.  fix this
    if (!synthFlag) {
	metaFlag = true;
    }

    log('count=' + countArg + ', max=' + maxArg);

    // TODO: add "show.js" with these exports
    if (showKnownArg) {
	if (!useClueList) {
	    // TODO: require max if !metaFlag
	    console.log('-u NAME:COUNT required with that option');
	    return 1;
	}
	if (!metaFlag) {
	    console.log('--meta required with that option');
	}
	if (!countArg) {
	    countArg = ClueManager.maxClues;
	}
	Show.compatibleKnownClues({
	    nameList: useClueList,
	    max:      countArg
	});
    }
    else if (Opt.options.test) {
	showValidSrcListCounts(Opt.options);
    }
    else if (showSourcesClueName) {
	showSources(showSourcesClueName);
    }
    else if (altSourcesArg || allAltSourcesFlag) {
	AltSources.show(allAltSourcesFlag ? {
	    all    : true,
	    output : outputArg,
	    count  : countArg
	} : {
	    all    : false,
	    name   : altSourcesArg,
	    output : outputArg
	});
    }
    else {
	doCombos({
	    sum:     countArg,
	    max:     maxArg,
	    require: requiredSizes,
	    sources: primarySourcesArg,
	    use:     useClueList
	});
    }
}

//

function loadClues(synthFlag, metaFlag, jsonArg,
		   validateAllOnLoad, ignoreLoadErrors) {
    var base;

    if ((synthFlag  || metaFlag) && jsonArg) {
	console.log('--json not allowed with --synth or --meta');
	return false;
    }
    if ((synthFlag !== undefined) && (metaFlag !== undefined)) {
	console.log('--synthesis and --meta not allowed');
	return false;
    }

    if (jsonArg) {
	base = jsonArg;
    }
    else if (!synthFlag) {
	base = 'meta';
    }
    else {
	base = 'synth';
    }

    log('loading all clues...');

    ClueManager.loadAllClues({
	baseDir:      base,
	validateAll:  validateAllOnLoad,
	ignoreErrors: ignoreLoadErrors
    });

    log('done.');

    return true;
}

//
//

function showValidSrcListCounts(options) {
    let srcList = options.test;
    var nameList;
    var countListArray;
    var count;
    var map;
    var resultList;

    if (!srcList) {
	throw new Error('missing arg, srcList: ' + srcList);
    }
    if (options.add && options.reject) {
	throw new Error('cannot specify both --add and --reject');
    }

    nameList = srcList.split(',');
    if (nameList.length < 2) {
//	throw new Error('use grep or supply 2 source names, dumbass');
    }
    nameList.sort();
    nameList.forEach(name => {
	console.log('name: ' + name);
    });

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    // each count list contains the clueMapArray indexes in which
    // each name appears
    countListArray = Array(_.size(nameList)).fill().map(() => []);
    //console.log(countListArray);
    for (count = 1; count <= ClueManager.maxClues; ++count) {
	map = ClueManager.knownClueMapArray[count];
	if (!_.isUndefined(map)) {
	    nameList.forEach((name, index) => {
		if (!_.isUndefined(map[name])) {
		    countListArray[index].push(count);
		}
	    });
	}
	else {
	    console.log('missing cluemap: ' + count);
	}
    }

    // verify that all names were found
    nameList.forEach((name, index) => {
	if (!countListArray[index]) {
	    throw new Error('Cannot find clue, ' + name +
			    ', array: ' + countListArray[index]);
	}
    });

    console.log(countListArray);

    resultList = Peco.makeNew({
	listArray: countListArray,
	max:       ClueManager.maxClues
    }).getCombinations();

    if (!resultList) {
	console.log('No matches');
	return;
    }
    let addCountSet = new Set();
    resultList.forEach(clueCountList => {
	let sum = 0;
	clueCountList.forEach(count => { sum += count });
	let sum2 = clueCountList.reduce((a, b) => a + b);
	if (sum !== sum2) {
	    throw new Error('I was wrong');
	}

	let result = Validator.validateSources({
	    sum:      sum,
	    nameList: nameList,
	    count:    nameList.length,
	    validateAll: true
	});
	//console.log('validate [' + nameList + ']: ' + result);
	let msg = clueCountList.toString();
	if (!result.success) {
	    msg += ': INVALID';
	}
	else {
	    let clueList = ClueManager.knownSourceMapArray[sum][nameList];
	    if (clueList) {
		msg += ': PRESENT as ' + _.toString(clueList.map(clue => clue.name));
	    }
	    else if (ClueManager.isRejectSource(nameList)) {
		msg += ': REJECTED';
	    }
	    else {
		// valid combo, neither known nor reject
		if (!_.isUndefined(options.add)) {
		    addCountSet.add(sum);
		}
	    }
	}
	console.log(msg);
    });

    if (!_.isUndefined(options.add)) {
	addCountSet.forEach(count => {
	    if (ClueManager.addClue(count, {
		name: options.add,
		src:  nameList.toString()
	    }, true));
	});
    }
    else if (!_.isUndefined(options.reject)) {
	ClueManager.addReject(nameList, true);
    }
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
    var comboListArray;
    var beginDate;
    var result;
    var count;
    var sourcesList;
    var set;

    if (args.sources) {
	args.sources = _.map(_.split(args.sources, ','), _.toNumber)
    }
    if (args.require) {
	args.require = _.map(_.split(args.require, ','), _.toNumber)
    }

    console.log('++combos' +
		', sum: ' + args.sum +
		', max: ' + args.max +
		', require: ' + args.require +
		', sources: ' + args.sources +
		', use: ' + args.use);

    beginDate = new Date();

    comboListArray = ComboMaker.makeCombos(args);

    log('--combos: ' + (new Duration(beginDate, new Date())).seconds + ' seconds');

    result = ClueManager.filter(comboListArray, args.sum);

    count = 0;
    _.keys(result.set).forEach(nameStr => {
	console.log(nameStr);
	++count;
    });

    console.log('total: ' + _.size(result.set) +
		' known: ' + result.known +
		' reject: ' + result.reject);

    if (result.array.length != count + result.known + result.reject) {
	//throw new Error('Amounts do not add up!');
    }

}

//
//

function showSources(clueName) {
    var result;
    var nc;
    var verbose;
    var clueSplitList = clueName.split(',');

    clueName = clueSplitList[0];
    nc = NameCount.makeNew(clueName);
    if (_.size(clueSplitList) > 1) {
	verbose = clueSplitList[1] == 'v';
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
	result.ncListArray.forEach(result => {
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

function showValidateResult(args) {
    var name;
    var map;

    if (args.result[args.key]) {
	console.log(args.title);
	map = args.result[args.key];
	for (name in map) {
	    map[name].forEach(ncList => {
		console.log(name + ': ' + ncList);
	    });
	}
    }
}

//

function setLogging(verboseArg) {
    var flag = Boolean(verboseArg);
    ClueManager.logging = flag;
    ComboMaker.logging  = flag;
    Validator.logging   = flag;
    AltSources.logging  = flag;
    ComboSearch.logging = flag;
    ResultMap.setLogging(flag);
    Peco.setLogging(flag);

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

var appBegin


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
