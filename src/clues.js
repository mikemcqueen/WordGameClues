//
// CLUES.JS
//

'use strict';

var _           = require('lodash');
var Duration    = require('duration');
var Np          = require('named-parameters');
var expect      = require('chai').expect;

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

// metamorphois -> synthesis -> harmonize -> finalize

var Opt = require('node-getopt')
    .create([
	['a', 'alt-sources=NAME',            'show alternate sources for the specified clue' ],
	['A', 'all-alt-sources',             'show alternate sources for all clues' ],
	['o', 'output',                      '  output json -or- clues(huh?)' ],
	['c', 'count=COUNT',                 '# of primary clues to combine' ],
	['d', 'allow-dupe-source',           'allow duplicate source, override default behavior of --meta' ],
	['f', 'final',                       'use final clues' ],
	['k', 'show-known',                  'show compatible known clues; -u <clue> required' ],
	['',  'csv',                         '  output in search-term csv format' ],
	['p', 'primary-sources=SOURCE[,SOURCE,...]', 'limit results to the specified primary source(s)' ],
	['q', 'require-counts=COUNT+',       'require clue(s) of specified count(s)' ],
	['r', 'harmony'            ,         'use harmony clues' ],
	['s', 'show-sources=NAME[:COUNT][,v]', 'show primary source combinations for the specified name[:count]' ],
	['t', 'test=SOURCE[,SOURCE,...]',    'test the specified source list, e.g. blue,fish' ],
	['',  'add=NAME',                    '  add combination to known list as NAME; use with --test' ],
	['',  'reject',                      '  add combination to reject list; use with --test' ],
	['u', 'use=NAME[:COUNT]+',           'use the specified name[:count](s)' ],
	['x', 'max=COUNT',                   'specify maximum # of components to combine'],
	['y', 'synthesis',                   'use synthesis clues' ],
	['z', 'flags=OPTION+',               'flags: 1=validateAllOnLoad,2=ignoreLoadErrors' ],

	['v', 'verbose=OPTION+',             'show logging. OPTION=load' ],
	['h', 'help',                        'this screen']
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
    let countArg = _.toNumber(Opt.options.count);
    let maxArg = _.toNumber(Opt.options.max);
    let useClueList = Opt.options.use;
    let verboseArg = Opt.options['verbose'];
    let showSourcesClueName = Opt.options['show-sources'];
    let altSourcesArg = Opt.options['alt-sources'];
    let allAltSourcesFlag = Opt.options['all-alt-sources'];
    let allowDupeSrcFlag = Opt.options['allow-dupe-source'];
    let showKnownArg = Opt.options['show-known'];
    let synthFlag = Opt.options['synthesis'];
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

    Validator.setAllowDupeFlags(synthFlag ? {
	allowDupeNameSrc: false,
	allowDupeSrc:     false, // NOTE: was true, i changed to false
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
    if (!loadClues(synthFlag, validateAllOnLoad, ignoreLoadErrors)) {
	return 1;
    }
    setLogging(verboseArg);

    // hacky.  fix this
    let metaFlag;
    if (!synthFlag) {
	metaFlag = true;
    }

    log('count=' + countArg + ', max=' + maxArg);

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
	if (!countArg) {
	    countArg = ClueManager.maxClues;
	}
	Show.compatibleKnownClues({
	    nameList: useClueList,
	    max:      countArg,
	    asCsv:    Opt.options.csv
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
	    output : Opt.options.output,
	    count  : countArg
	} : {
	    all    : false,
	    name   : altSourcesArg,
	    output : Opt.options.output
	});
    }
    else {
	doCombos({
	    sum:     countArg,
	    max:     maxArg,
	    require: Opt.options['require-counts'],
	    sources: Opt.options['primary-sources'],
	    use:     useClueList
	});
    }
}

//

function loadClues (synthFlag, validateAllOnLoad, ignoreLoadErrors) {
    let base = 'meta';
    if (synthFlag) {
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
    expect(options.test, 'options.test').to.be.a('string');
    if (options.reject) {
	expect(options.add, 'cannot specify both --add and --reject').to.be.undefined;
    }

    let nameList = options.test.split(',').sort();
    nameList.forEach(name => {
	console.log('name: ' + name);
    });

    /// TODO, check if existing sourcelist (knownSourceMapArray)

    // each count list contains the clueMapArray indexes in which
    // each name appears
    let countListArray = Array(_.size(nameList)).fill().map(() => []);
    //console.log(countListArray);
    for (let count = 1; count <= ClueManager.maxClues; ++count) {
	let map = ClueManager.knownClueMapArray[count];
	if (!_.isUndefined(map)) {
	    nameList.forEach((name, index) => {
		if (_.has(map, name)) {
		    countListArray[index].push(count);
		}
	    });
	}
	else {
	    console.log('missing known cluemap #' + count);
	}
    }

    // verify that all names were found
    nameList.forEach((name, index) => {
	expect(countListArray[index], `cannot find clue, ${name}`).to.exit;
    });

    console.log(countListArray);

    let resultList = Peco.makeNew({
	listArray: countListArray,
	max:       ClueManager.maxClues
    }).getCombinations();
    if (_.isEmpty(resultList)) {
	console.log('No matches');
	return;
    }

    let addCountSet = new Set();
    let known = false;
    let reject = false;
    resultList.forEach(clueCountList => {
	let sum = clueCountList.reduce((a, b) => a + b);
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
	else if (ClueManager.isRejectSource(nameList)) {
	    msg += ': REJECTED';
	    reject = true;
	}
	else {
	    if (nameList.length === 1) {
		let name = nameList[0];
		let nameSrcList = ClueManager.clueListArray[sum]
		    .filter(clue => clue.name === name)
		    .map(clue => clue.src);

		if (nameSrcList.length > 0) {

		//let clueNameList = ClueManager.clueListArray[sum].map(clue => clue.name);
		//if (clueNameList.includes(name)) {
		//
		    
		    /*
		    ClueManager.clueListArray[sum].forEach(clue => {
			if (clue.name === name) {
			    clueSrcList.push(`"${clue.src}"`);
			}
		    });
		    */
		    msg += ': PRESENT as clue with sources: ' + nameSrcList.join(' - ');
		}
	    }
	    else {
		let clueList = ClueManager.knownSourceMapArray[sum][nameList];
		if (!_.isUndefined(clueList)) {
		    msg += ': PRESENT as ' + clueList.map(clue => clue.name);
		    known = true;
		}
		if (options.add) {
		    addCountSet.add(sum);
		}
	    }
	}
	console.log(msg);
    });

    if (!_.isUndefined(options.add)) {
	if (nameList.length === 1) {
	    console.log('WARNING! ignoring --add due to single source');
	}
	else if (reject) {
	    console.log('WARNING! cannot add known clue: already rejected, ' + nameList);
	}
	else {
	    addClues(addCountSet, options.add, nameList.toString());
	}
    }
    else if (options.reject) {
	if (nameList.length === 1) {
	    console.log('WARNING! ignoring --reject due to single source');
	}
	else if (known) {
	    console.log('WARNING! cannot add reject clue: already known, ' + nameList);
	}
	else {
	    addReject(nameList.toString());
	}
    }
}

//

function addClues(countSet, name, src) {
    expect(countSet).to.be.a('Set');
    expect(name).to.be.a('string');
    expect(src).to.be.a('string');
    countSet.forEach(count => {
	if (ClueManager.addClue(count, {
	    name: name,
	    src:  src
	}, true)) {
	    console.log('updated ' + count);
	}
	else {
	    console.log('update of ' + count + ' failed.');
	}
    });
}

//

function addReject(nameList) {
    if (ClueManager.addReject(nameList, true)) {
	console.log('updated');
    }
    else {
	console.log('update failed');
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
    if (!_.isUndefined(args.sources)) {
	args.sources = _.chain(args.sources).split(',').map(_.toNumber).value();
    }
    if (!_.isUndefined(args.require)) {
	args.require = _.chain(args.require).split(',').map(_.toNumber).value();
    }

    console.log('++combos' +
		', sum: ' + args.sum +
		', max: ' + args.max +
		', require: ' + args.require +
		', sources: ' + args.sources +
		', use: ' + args.use);

    let beginDate = new Date();
    let nameCsvList = ComboMaker.makeCombos(args);
    log('--combos: ' + (new Duration(beginDate, new Date())).seconds + ' seconds');

    let total = nameCsvList.length;
    let result = ClueManager.filter(nameCsvList, args.sum);

    _.keys(result.set).forEach(nameCsv => console.log(nameCsv));

    console.log(`total: ${total}` +
		', filtered: ' + _.size(result.set) +
		', known: ' + result.known +
		', reject: ' + result.reject +
		', duplicate: ' + result.duplicate);

    if (nameCsvList.length !== _.size(result.set) + result.known + result.reject + result.duplicate) {
	console.log('WARNING: amounts to not add up!');
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
