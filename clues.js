'use strict';

var Duration    = require('duration');

var ClueManager = require('./clue_manager');
var ComboMaker  = require('./combo_maker');
var ComboSearch = require('./combo_search');
var Validator   = require('./validator');
var ClueList    = require('./clue_list');
var NameCount   = require('./name_count');
var Peco        = require('./peco');

var QUIET = false;

// initialize command line options.  do this before logger.
//

var Opt = require('node-getopt')
    .create([
	['a' , 'show-alternates=ARG',         'show alternate sources for the specified clue' ],
	['A' , 'show-all-alternates',         'show alternate sources for all clues' ],
	['c' , 'count=ARG'          ,         '# of primary clues to combine' ], 
	['d' , 'allow-dupe-source'  ,         'allow duplicate source, override default behavior of --meta' ],
	['m' , 'meta'               ,         'use metamorphosis clues' ],
	['o' , 'output'         ,         'output: json -or- clues' ],
	['r' , 'require=ARG+'       ,         'require compound clue(s) of specified count(s)' ],
	['s' , 'show-sources=ARG'   ,         'show possible source combinations for the specified clue' ],
	['t' , 'test=ARG'           ,         'test the specified source list, e.g. blue,fish' ],
	['u' , 'use=ARG+'           ,         'use the specified clue[:count]' ],
	['x' , 'max=ARG'            ,         'specify maximum # of components to combine'],
	
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
    var count;
    var max;
    var requiredSizes;
    var metaFlag;
    var verboseFlag;
    var showSourcesClueName;
    var showAlternatesArg;
    var showAllAlternatesArg;
    var testSrcList;
    var useClueList;
    var allowDupeSrcFlag;
    var needCount;
    var outputArg;

/*    if (Opt.argv.length) {
	console.log('Usage: node clues.js [options]');
	console.log('');
	console.log(Opt);
	return 1;
    }
*/

    // options

    count = Opt.options['count'];
    max = Opt.options['max'];
    requiredSizes = Opt.options['require'];
    useClueList = Opt.options['use'];
    metaFlag = Opt.options['meta'];
    verboseFlag = Opt.options['verbose'];
    outputArg = Opt.options['output'];
    showSourcesClueName = Opt.options['show-sources'];
    showAlternatesArg = Opt.options['show-alternates'];
    showAllAlternatesArg = Opt.options['show-all-alternates'];
    allowDupeSrcFlag = Opt.options['allow-dupe-source'];
    testSrcList = Opt.options['test'];

    if (!max) {
	max = 2; // TODO: default values in opt
    }
    if (!count) {
	needCount = true;
	
	if (showSourcesClueName ||
	    showAlternatesArg ||
	    showAllAlternatesFlag ||
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
	//console.log ('invalid --output option, ' + outputArg);
    }

    ClueManager.logging = verboseFlag;
    ComboMaker.logging  = verboseFlag;
    Validator.logging   = verboseFlag;
    ComboSearch.logging = verboseFlag;
    LOGGING = verboseFlag;

    Validator.setAllowDupeFlags(metaFlag ? {
	allowDupeNameSrc: false,
	allowDupeSrc:     allowDupeSrcFlag ? true : false,
	allowDupeName:    true,
    } : {
	allowDupeNameSrc: false,
	allowDupeSrc:     true,
	allowDupeName:    true,
    });

    //

    if (LOGGING) {
	log('count=' + count + ' max=' + max);
    }

    ClueManager.loadAllClues(metaFlag ? {
	known:    'meta',
	reject:   'metarejects',
	max:      MAX_META_CLUE_COUNT,
	required: REQ_META_CLUE_COUNT
    } : {
	known:    'clues',
	reject:   'rejects',
	max:      MAX_SYNTH_CLUE_COUNT,
	required: REQ_SYNTH_CLUE_COUNT
    });

    if (LOGGING) {
	log('loaded...');
    }

    if (testSrcList) {
	showValidSrcListCounts(testSrcList, metaFlag ? 
			       MAX_META_CLUE_COUNT : MAX_SYTH_CLUE_COUNT);
    }
    else if (showSourcesClueName) {
	showSources(showSourcesClueName);
    }
    else if (showAlternatesArg || showAllAlternatesArg) {
	showAlternates(showAllAlternatesArg ? {
	    all    : true
	} : {
	    all    : false,
	    name   : showAlternatesArg,
	    output : outputArg
	});
    }
    else {
	doCombos({
	    sum:     count,
	    max:     max,
            require: requiredSizes,
	    use:     useClueList,
	});
    }
}

//
//

function showValidSrcListCounts(srcList, max) {
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
	throw new Error('use grep or supply 2 source names, dumbass');
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
    }

    // verify that all names were found
    nameList.forEach((name, index) => {
	if (!countListArray[index]) {
	    throw new Error('Cannot find clue, ' + name);
	}
    });

    resultList = (new Peco({
	max: ClueManager.maxClues,
	listArray: countListArray
    })).getCombinations();

    if (resultList.length) {
	resultList.forEach(clueCountList => {
	    console.log(clueCountList);
	});
    }
    else {
	console.log('No matches');
    }
}

//
//

function doCombos(args) {
    var comboListArray;
    var beginDate;
    var result;
    var count;

    log('++combos');

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
    var nc = new NameCount(clueName);
    nc.log();

    if (!nc.count) {
	throw new Error('Need to supply a count as name:count (for now)');
    }
    
    log('++sources');

    Validator.validateSources({
	sum:      nc.count,
	nameList: [ nc.name ],
	count:    1, // nc.count, ???
	showAll:  true
    });
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

	name = argList[0];
	var nc = new NameCount(name);
	
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
	console.log('showAlternates: all');

	for (count = 2; count <= ClueManager.maxClues; ++count) {
	    map = ClueManager.knownClueMapArray[count];
	    for (name in map) {
		displayAllAlternates(name, ComboSearch.findAlternateSourcesForName(
		    new NameCount(name, count).toString()));
	    }
	}
    }
}

//

function displayAllAlternates(name, ncListArrayArray, count, output) {
    var ncListArray;

    if (!ncListArrayArray.length) {
	return;
    }

    if (output && count) {
	displayModifiedClueList(name, count, ncListArrayArray[count])
    }
    else if (count) {
	ncListArray = ncListArrayArray[count];
	displayAlternate(name, count, ncListArray);
    }
    else {
	ncListArrayArray.forEach((ncListArray, index) => {
	    displayAlternate(name, index, ncListArray);
	});
    }
}

//

function displayAlternate(name, count, ncListArray) {
    var s;
    var nameList;
    var found = false;

    s = name + '[' + count + '] ';
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

function displayModifiedClueList(name, count, ncListArray) {
    var clueList = ClueManager.clueListArray[count];
    var clue;

    // no loop here because entries will always have the
    // same sources, so just add the first one
    //ncListArray.forEach((ncList, nclaIndex) => {
    clue = { name: name, src: [] };
    ncListArray[0].forEach(nc => {
	clue.src.push(nc.name);
    });
    clueList.push(clue);
    //});

    console.log(clueList.toJSON())
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
    if (!QUIET) {
	console.log('runtime: ' + (new Duration(appBegin, new Date())).seconds + ' seconds');
    }
}
