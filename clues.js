'use strict';

var _           = require('lodash');
var Duration    = require('duration');
var Np          = require('named-parameters');

var ClueManager = require('./clue_manager');
var ComboMaker  = require('./combo_maker');
var ComboSearch = require('./combo_search');
var Validator   = require('./validator');
var ClueList    = require('./clue_list');
var NameCount   = require('./name_count');
var Peco        = require('./peco');
var Show        = require('./show');

var QUIET = false;

// initialize command line options.  do this before logger.
//

var Opt = require('node-getopt')
    .create([
	['a' , 'show-alternates=NAME',        'show alternate sources for the specified clue' ],
	['A' , 'show-all-alternates',         'show alternate sources for all clues' ],
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

	['v' , 'verbose'            ,         'show debug output' ],
	['h' , 'help'               ,         'this screen']
    ])
    .bindHelp().parseSystem();

var LOGGING = false;

var MAX_SYNTH_CLUE_COUNT  = 25;
var REQ_SYNTH_CLUE_COUNT  = 11; // should make this 12 soon
var MAX_META_CLUE_COUNT   = 9;
var REQ_META_CLUE_COUNT   = 9;

//
//

function main() {
    var countArg;
    var maxArg;
    var requiredSizes;
    var metaFlag;
    var synthFlag;
    var verboseFlag;
    var showSourcesClueName;
    var showAlternatesArg;
    var showAllAlternatesArg;
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
    verboseFlag = Opt.options['verbose'];
    outputArg = Opt.options['output'];
    showSourcesClueName = Opt.options['show-sources'];
    showAlternatesArg = Opt.options['show-alternates'];
    showAllAlternatesArg = Opt.options['show-all-alternates'];
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
	    showAlternatesArg ||
	    showAllAlternatesArg ||
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

    ClueManager.logging = verboseFlag;
    ComboMaker.logging  = verboseFlag;
    Validator.logging   = verboseFlag;
    ComboSearch.logging = verboseFlag;
    LOGGING = verboseFlag;

    Validator.setAllowDupeFlags(synthFlag ? {
	allowDupeNameSrc: false,
	allowDupeSrc:     true,
	allowDupeName:    true,
    } : {
	allowDupeNameSrc: false,
	allowDupeSrc:     allowDupeSrcFlag ? true : false,
	allowDupeName:    true,
    });

    //
    if (!loadClues(synthFlag, metaFlag, jsonArg)) {
	return 1;
    }

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
    else if (showAlternatesArg || showAllAlternatesArg) {
	showAlternates(showAllAlternatesArg ? {
	    all    : true,
	    output : outputArg,
	    count  : countArg
	} : {
	    all    : false,
	    name   : showAlternatesArg,
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

    resultList = (new Peco({
	listArray: countListArray,
	max:       ClueManager.maxClues
    })).getCombinations();

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
	    result: result,
	    key:    Validator.STAGING_KEY
	});
	showValidateResult({
	    title:  'compound',
	    result: result,
	    key:    Validator.COMPOUND_KEY
	});
	showValidateResult({
	    title: 'primary',
	    result: result,
	    key:    Validator.PRIMARY_KEY
	});
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
//

function showAlternates(args) {
    var count;
    var map;
    var name;
    var count;
    var argList;
    var ncListArray;
    var max;
    var results;
    var allAlternates;

    /*
    if (!args.all) {
	number = Number(args.name);
	if ((number >= 2) && (number <= ClueManager.maxClues)) {
	    // treat this as "show all" for count: number
	    args.all = true;
	}
    }
    */

    if (!args.all) {
	argList = args.name.split(',');
	if (argList.length > 1) {
	    count = argList[1];
	}
	if (args.output && !count) {
	    console.log('WARNING: output format ignored, no ",count" specified');
	}

	name = argList[0]; // name:count
	var nc = NameCount.makeNew(name);

	if (!nc.count) {
	    throw new Error('Need to supply a count as name:count (for now)');
	}
	if (!args.output) {
	    console.log('showAlternates: ' + nc);
	}
	ncListArray = ComboSearch.findAlternateSourcesForName(name, count);
	displayAllAlternates(nc.name, ncListArray, count, args.output);
    }
    else {
	if (args.output && !args.count) {
	    console.log('WARNING: output format ignored, no -c COUNT specified');
	}
	if (!args.output) {
	    console.log('showAlternates: all');
	}

	count = 2;
	max = ClueManager.maxClues;

	allAlternates = [];
	for (; count <= max; ++count) {
	    map = ClueManager.knownClueMapArray[count];
	    for (name in map) {
		//console.log(name +':' + count);
		ncListArray = ComboSearch.findAlternateSourcesForName(NameCount.makeNew(name, count).toString());
		displayAllAlternates(name, ncListArray, args.count, args.output, true, allAlternates);
	    }
	}
	if (args.output && args.count) {
	    console.log(ClueManager.clueListArray[args.count].toJSON());
	    //displayModifiedClueListList(args.count, allAlternates);
	}

    }
}

//
// 
function displayAllAlternates(name, ncListArrayArray, count, output, all, allAlternates) {
    var ncListArray;
    var clue;

    if (!ncListArrayArray.length) {
	return;
    }

//    console.log(name + ' : ' + count + ' : ' + output);

    if (!all) {
	if (output && count) {
	    displayModifiedClueList(count, getAlternateClue(name, ncListArrayArray[count]))
	}
	else if (count) {
	    ncListArray = ncListArrayArray[count];
	    displayAlternate(name, count, ncListArray);
	}
    }
    else {
	ncListArrayArray.forEach((ncListArray, index) => {
	    if (!output || !count) {
		if (count) {
		    if ((count === index)) {
			displayAlternate(name, index, ncListArray);
		    }
		}
		else {
		    displayAlternate(name, index, ncListArray);
		}
	    }
	    else {
		if (count === index) {
		    clue = getAlternateClue(name, ncListArray);
		    //console.log ('ADDING: name: ' + clue.name + ', src: ' + clue.src);
		    //allAlternates.push(clue);
		    ClueManager.addClue(count, clue);
		}
	    }
	});
    }
}

//

function displayAlternate(name, count, ncListArray) {
    var s;
    var nameList;
    var found = false;

    s = name + '[' + count + '] ';
    s += format2(s, 20) + ' ';
    ncListArray.forEach((ncList, nclaIndex) => {
	nameList = [];
	ncList.forEach(nc => {
	    nameList.push(nc.name);
	});
	nameList.sort();
	if (ClueManager.knownSourceMapArray[count][nameList.toString()]) {
	    //console.log('found: ' + nameList + ' in ' + count);
	    return; // continue
	}

	if (found) {
	    s += ', ';
	}
	ncList.forEach((nc, nclIndex) => {
	    if (nclIndex > 0) {
		s += ' ';
	    }
	    s += nc;
	});
	found = true;
    });
    if (found) {
	console.log(s);
    }
}

//

function format2(text, span)
{
    var result = "";
    for (var len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//

function displayModifiedClueList(count, clue) {
    var clueList = ClueManager.clueListArray[count];
    
    clueList.push(clue);
    
    console.log(clueList.toJSON());
}

function displayModifiedClueListList(count, alternateClues) {
    var clueList = ClueManager.clueListArray[count];
    
    alternateClues.forEach(clue => {
	clueList.push(clue);
    });
    
    console.log(clueList.toJSON());
}


//

function getAlternateClue(name,  ncListArray) {
    var clue;
    // no loop here because entries will always have the
    // same sources, so just add the first one
    clue = { name: name, src: [] };
    ncListArray[0].forEach(nc => {
	clue.src.push(nc.name);
    });
    clue.src = _.toString(clue.src);
    return clue;
}

//
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
