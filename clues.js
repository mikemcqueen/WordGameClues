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

// initialize command line options.  do this before logger.
//

var Opt = require('node-getopt')
    .create([
	['a' , 'alt-sources=NAME',            'show alternate sources for the specified clue' ],
	['A' , 'all-alt-sources',             'show alternate sources for all clues' ],
	['c' , 'count=COUNT',                 '# of primary clues to combine' ],
	['d' , 'allow-dupe-source'  ,         'allow duplicate source, override default behavior of --meta' ],
	[''  , 'json=FILEBASE',               'specify base filename of clue files' ],
	['k' , 'show-known'         ,         'show compatible known clues; -u <clue> required' ],
	['m' , 'meta'               ,         'use metamorphosis clues, same as --json meta (default)' ],
	['o' , 'output'             ,         'output json -or- clues' ],
	['p' , 'primary-sources=SOURCE[,SOURCE,...]', 'limit results to the specified primary source(s)' ],
	['q' , 'require-counts=COUNT+',       'require clue(s) of specified count(s)' ],
	['s' , 'show-sources=NAME[:COUNT]',   'show possible source combinations for the specified name[:count]' ],
	['t' , 'test=NAME[,NAME,...]',        'test the specified source list, e.g. blue,fish' ],
	['u' , 'use=NAME[:COUNT]+',           'use the specified name[:count](s)' ],
	['x' , 'max=COUNT',                   'specify maximum # of components to combine'],
	['y' , 'synthesis',                   'use synthesis clues, same as --json clues' ],

	['v' , 'verbose=OPTION+',             'show logging. OPTIONS=load' ],
	['h' , 'help'               ,         'this screen']
    ])
    .bindHelp().parseSystem();

//

var LOGGING = false;
var QUIET = false;

//

const VERBOSE_FLAG_LOAD     = 'load';

const MAX_SYNTH_CLUE_COUNT  = 25;
const REQ_SYNTH_CLUE_COUNT  = 11; // should make this 12 soon
const MAX_META_CLUE_COUNT   = 9;
const REQ_META_CLUE_COUNT   = 9;

//
//

function main() {
    var countArg;
    var maxArg;
    var requiredSizes;
    var metaFlag;
    var synthFlag;
    var verboseArg;
    var showSourcesClueName;
    var altSourcesArg;
    var allAltSourcesFlag;
    var testSrcList;
    var useClueList;
    var allowDupeSrcFlag;
    var needCount;
    var outputArg;
    var showKnownArg;
    var jsonArg;
    var primarySourcesArg;

/*    if (Opt.argv.length) {
	console.log('Usage: node clues.js [options]');
	console.log('');
	console.log(Opt);
	return 1;
    }
*/

    // options

    countArg = _.toNumber(Opt.options['count']);
    maxArg = _.toNumber(Opt.options['max']);
    requiredSizes = Opt.options['require-counts'];
    useClueList = Opt.options['use'];
    metaFlag = Opt.options['meta'];
    verboseArg = Opt.options['verbose'];
    outputArg = Opt.options['output'];
    showSourcesClueName = Opt.options['show-sources'];
    altSourcesArg = Opt.options['alt-sources'];
    allAltSourcesFlag = Opt.options['all-alt-sources'];
    allowDupeSrcFlag = Opt.options['allow-dupe-source'];
    testSrcList = Opt.options['test'];
    showKnownArg = Opt.options['show-known'];
    synthFlag = Opt.options['synthesis'];
    jsonArg = Opt.options['json'];
    primarySourcesArg = Opt.options['primary-sources'];

    if (!maxArg) {
	maxArg = 2; // TODO: default values in opt
    }
    if (!countArg) {
	needCount = true;

	if (showSourcesClueName ||
	    altSourcesArg ||
	    allAltSourcesFlag ||
	    showKnownArg ||
	    testSrcList ||
	    useClueList)
	{
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
	allowDupeSrc:     allowDupeSrcFlag ? true : false,
	allowDupeName:    true,
    });

    setLogging(_.includes(verboseArg, VERBOSE_FLAG_LOAD));

    //
    if (!loadClues(synthFlag, metaFlag, jsonArg)) {
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
    else if (testSrcList) {
	showValidSrcListCounts(testSrcList);
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

function loadClues(synthFlag, metaFlag, jsonArg) {
    var base;
    var max;
    var required;

    if ((synthFlag  || metaFlag) && jsonArg) {
	console.log('--json not allowed with --synth or --meta not allowed');
	return false;
    }
    if ((synthFlag != undefined) && (metaFlag != undefined)) {
	console.log('--synthesis and --meta not allowed');
	return false;
    }

    // default to meta sizes, unless --synthesis. good enough for now
    max =      MAX_META_CLUE_COUNT;
    required = REQ_META_CLUE_COUNT;

    if (jsonArg) {
	base = jsonArg;
    }
    else if (!synthFlag) {
	base = 'meta';
    }
    else {
	max =      MAX_SYNTH_CLUE_COUNT;
	required = REQ_SYNTH_CLUE_COUNT;
	base = 'clues';
    }

    log('loading all clues...');

    ClueManager.loadAllClues({
	known:    base,
	reject:   base + 'rejects',
	max:      max,
	required: required
    });

    log('done.');

    return true;
}

//
//

function showValidSrcListCounts(srcList) {
    var nameList;
    var countListArray;
    var count;
    var map;
    var resultList;

    if (!srcList) {
	throw new Error('missing arg, srcList: ' + srcList);
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
    countListArray = [];
    for (count = 1; count <= ClueManager.maxClues; ++count) {
	map = ClueManager.knownClueMapArray[count];
	if (map) {
	    nameList.forEach((name, index) => {
		if (map[name]) {
		    if (!countListArray[index])  {
			countListArray[index] = [ count ];
		    }
		    else {
			countListArray[index].push(count);
		    }
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
    resultList.forEach(clueCountList => {
	var sum;
	var result;
	var msg;
	var clueList;
	sum = 0;
	clueCountList.forEach(count => { sum += count });
	result = Validator.validateSources({
	    sum:      sum,
	    nameList: nameList,
	    count:    nameList.length
	});
	//console.log('validate [' + nameList + ']: ' + result);
	msg = clueCountList.toString();
	if (!result) {
	    msg += ': INVALID';
	}
	else {
	    clueList = ClueManager.knownSourceMapArray[sum][nameList];
	    if (clueList) {
		msg += ': PRESENT as ';
		clueList.forEach((clue, index) => {
		    if (index > 0) {
			msg += ', ';
		    }
		    msg += clue.name;
		});
	    }

	}
	console.log(msg);
    });
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
    result.array.forEach(function(clueList) {
	clueList.display();
	++count;
    });

    console.log('total: ' + result.array.length +
		' filtered: ' + count +
		' known: ' + result.known +
		' reject: ' + result.reject);

    if (result.array.length != count + result.known + result.reject) {
	throw new Error('Amounts do not add up!');
    }

}

//
//

function showSources(clueName) {
    var result;
    var nc = NameCount.makeNew(clueName);
    nc.log();

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
    if (result) {
	showValidateResult({
	    title:  'staging',
	    result: result.resultMap,
	    key:    Validator.STAGING_KEY
	});
	showValidateResult({
	    title:  'compound',
	    result: result.resultMap,
	    key:    Validator.COMPOUND_KEY
	});
	showValidateResult({
	    title: 'primary',
	    result: result.resultMap,
	    key:    Validator.PRIMARY_KEY
	});
	Validator.dumpResultMap(result.vsResultMap);
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
