//
// clues.js
//

'use strict';

const _           = require('lodash');
const AltSources  = require('../modules/alt-sources');
const ClueList    = require('../types/clue-list');
const ClueManager = require('../modules/clue-manager');
const Clues       = require('../modules/clue-types');
const ComboMaker  = require('../modules/combo-maker');
const ComboSearch = require('../modules/combo-search');
const Components  = require('../modules/show-components');
const Debug       = require('debug')('clues');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Log         = require('../modules/log')('clues');
const NameCount   = require('../types/name-count');
const Note        = require('../modules/note');
const Opt         = require('node-getopt');
const Peco        = require('../modules/peco');
const PrettyMs    = require('pretty-ms');
const Show        = require('../modules/show');
const Stringify   = require('stringify-object');
const ResultMap   = require('../types/result-map');
const Timing      = require('debug')('timing');
const Validator   = require('../modules/validator');

// initialize command line options.  do this before logger.
//

// TODO:
// solve the -p/-y problem: standardize how to set/initialize state based on target clues
// 

const CmdLineOptions = Opt.create(_.concat(Clues.Options, [
    ['a', 'alt-sources=NAME',                  'show alternate sources for the specified clue NAME'],
    ['A', 'all-alt-sources',                   'show alternate sources for all clues'],
    ['o', 'output',                            '  output json -or- clues(huh?)'],
    ['c', 'count=COUNT[LO,COUNTHI]',           'show combos of the specified COUNT; if COUNTHI, treat as range'],
    ['x', 'max=COUNT',                         '  maximum # of sources to combine'],
    ['',  'primary',                           '  show combos as primary source clues' ],
    ['',  'copy-from=SOURCE',                  'copy clues from source cluetype; e.g. p1.1'],
    ['',  'save',                              '  save clue files'],
    ['',  'allow-dupe-source',                 '  allow duplicate sources'],
    ['',  'merge-style',                       '  merge-style, no validation except for immediate sources'],
    ['',  'remaining',                         '  only word combos not present in any named note'],
    ['q', 'require-counts=COUNT+',             '  require clue(s) of specified COUNT(s)' ],
    ['i', 'primary-sources=SOURCE[,SOURCE,...]','limit results to the specified primary SOURCE(s)'],
    ['',  'inverse',                           '  or the inverse of those source(s); use with -i'],
    ['k', 'show-known',                        'show compatible known clues; -u <clue> required' ],
    ['',  'csv',                               '  output in search-term csv format' ],
    ['',  'files',                             '  output in result file full-path format' ],
    ['s', 'show-sources=NAME[:COUNT][,v]',     'show primary source combos for the specified NAME[:COUNT]' ],
    ['t', 'test=SOURCE[,SOURCE,...]',          'test the specified source list, e.g. blue,fish' ],
    ['',  'add=NAME',                          '  add combination to known list as NAME; use with --test' ],
    ['',  'remove=NAME',                       '  remove combination from known list as NAME; use with --test' ],
    ['',  'reject',                            '  add combination to reject list; use with --test' ],
    ['',  'fast',                              '  use fast method' ],
    ['',  'validate',                          '  treat SOURCE as filename, validate all source lists in file'],
    ['',  'combos',                            '    validate all combos of sources/source lists in file'],
    ['u', 'use=NAME[:COUNT]+',                 'use the specified NAME[:COUNT](s)' ],
    ['',  'allow-used',                        '  allow used clues in clue combo generation' ],
    ['',  'production',                        'use production note store'],
    ['z', 'flags=OPTION+',                     'flags: 2=ignoreLoadErrors' ],
    ['v', 'verbose',                           'more output'],
    ['h', 'help',                              'this screen']
])).bindHelp();

//

let LOGGING = false;
let QUIET = false;

//

const VERBOSE_FLAG_LOAD     = 'load';

//

function usage (msg) {
    console.log(msg + '\n');
    CmdLineOptions.showHelp();
    process.exit(-1);
}

//

function loadClues (clues, ignoreLoadErrors) {
    log('loading all clues...');
    ClueManager.loadAllClues({
        clues,
        validateAll:  true,
        ignoreErrors: ignoreLoadErrors
    });
    log('done.');
    return true;
}

function convertUseToPrimarySources (args) {
    // build ncList of supplied name:counts
    const ncList = args.use.map(ncStr => NameCount.makeNew(ncStr));
    for (const nc of ncList) {
        if (!nc.count || _.isNaN(nc.count)) {
            console.log('All -u names require a count (for now)');
            return { success: false };
        }
    }

    // TODO: some more clear way to extract just ".count"s into an array, then sum them
    const sum = ncList.reduce((a, b) => Object({ count: (a.count + b.count) })).count;
    const remain = ClueManager.maxClues - sum;
    if (remain < 1) {
        console.log(`The sum of the specified clue counts (${sum})` +
                    ` equals or exceeds the maximum clue count (${ClueManager.maxClues})`);
        return { success: false };
    }
    
    Debug('convertNcStrToPrimarySources ' + args.use + 
                ', sum: ' + sum + ', remain: ' + remain);

    // first, make sure the supplied nameList:sum by itself is a valid clue
    // combination, and find out how many primary-clue variations there
    // are in which the clue names in args.nameList exist.
    let vsResult = Validator.validateSources({
        sum:         sum,
        nameList:    ncList.map(nc => nc.name),
        count:       args.use.length,
        validateAll: true
    });
    if (!vsResult.success) {
        console.log(`The ncStr [${args.use}] is not a valid clue combination`);
        return { success: false };
    }

    console.log(`results: ${vsResult.list.length}`);

    // TODO: for each primary-clue variation from validateResults
    const nameSrcList = vsResult.list[0].nameSrcList;
    return {
        success: true,
        sources: nameSrcList.map(nc => _.toString(nc.count)),
        clues:   nameSrcList.map(nc => _.toString(nc.name))
    };
}

//
// args:
//  sum:     countArg,
//  max:     maxArg,
//  require: requiredSizes,
//  sources: primarySourcesArg,
//  use:     useClueList
//

function doCombos(args, options) {
    if (!_.isUndefined(args.sources)) {
        args.sources = _.chain(args.sources).split(',').map(_.toNumber).value();
    }
    if (!_.isUndefined(args.require)) {
	// is _chain even necessary here?
        args.require = _.chain(args.require).split(',').map(_.toNumber).value();
    }
    let sumRange;
    if (!_.isUndefined(args.sum)) {
	// is _chain even necessary here?
        sumRange = _.chain(args.sum).split(',').map(_.toNumber).value();
    }
    Expect(sumRange).is.an.Array().with.property('length').below(3); // of.at.most(2);
    Debug('++combos' +
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
        // TODO: return # of combos filtered due to note name match
        const comboList = ComboMaker.makeCombos(args, options);
        args.max = max;
        total += comboList.length;
        const filterResult = ClueManager.filter(comboList, args.sum, comboMap);
        known += filterResult.known;
        reject += filterResult.reject;
        duplicate += filterResult.duplicate;
    }
    let d = new Duration(beginDate, new Date()).milliseconds;
    _.keys(comboMap).forEach(nameCsv => console.log(nameCsv));
    //console.log(`${Stringify(comboMap)}`);
    
    Debug(`total: ${total}` +
                ', filtered: ' + _.size(comboMap) +
                ', known: ' + known +
                ', reject: ' + reject +
                ', duplicate: ' + duplicate);
    Debug(`--combos: ${PrettyMs(d)}`);

    if (total !== _.size(comboMap) + known + reject + duplicate) {
        Debug('WARNING: amounts to not add up!');
    }
}

//

async function getNamedNoteNames(options) {
    if (options.production) Log.info('---PRODUCTION---');

    return Note.getSomeMetadata(options)
        .filter(metadata => Note.chooser(metadata, options))
        .then(metadataList => {
            if (_.isEmpty(metadataList)) usage('no notes found');
            Log.info(`found ${metadataList.length} notes`);
            return metadataList;
        }).map(metadata => {
            Log.info(`meta note: ${metadata.title}`);
            const index = _.lastIndexOf(metadata.title, '.') + 1;
            if (index === 0 || index === metadata.title.length) {
                usage(`invalid named note: ${metadata.title}`);
            }
            return metadata.title.slice(index, metadata.title.length);
        });
}

//

function combo_maker(args, options) {
    return Promise.resolve(options.remaining ? getNamedNoteNames(options) : false)
        .then(noteNames => {
            if (noteNames) {
                Log.info(`note names: ${noteNames}`);
                options.note_names = noteNames;
            }
            return doCombos(args, options);
        });
}

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

    const entries = ClueManager.getKnownSourceMapEntries(nc);
    if (!entries) {
        usage('explosion');
    }
    entries.forEach(entry => {
        entry.results.forEach(result => {
            console.log(Stringify(result.resultMap.map(), { indent: '  ' }));
            console.log('nameSrcList: ' + result.nameSrcList);
            if (verbose) {
                console.log('ncList:      ' + result.ncList);
                result.resultMap.dump();
            }
        });
    });
}

//

function copyClues (fromType, options = {}) {
    const dir = fromType.baseDir;
    let total = 0;
    let copied = 0;
    for (let count = 2; count < 20; ++count) {
	let list;
	try {
	    list = ClueManager.loadClueList(count, { dir });
	} catch (err) {
	    if (!_.includes(err.message, 'ENOENT')) throw err;
	    continue;
	}
	total += _.size(list);
	for (let clue of list) {
	    const nameList = clue.src.split(',').sort();
	    const result = ClueManager.getCountListArrays(clue.src, { add: true });
	    if (!result) {
		Debug(`No matches for: ${clue.src}`);
		continue;
	    }
	    Debug(`Adding ${clue.name}:${clue.src}, set: ${Stringify(result.addRemoveSet)}`);
	    const count = ClueManager.addRemoveOrReject({
		add: clue.name,
		isReject: !_.isEmpty(result.reject)
	    }, nameList, result.addRemoveSet, { save: options.save });
	    copied += count;
	}
    }
    Log.info(`total: ${total}, copied: ${copied}`);
}

//

function setLogging (flag) {
    ClueManager.logging = flag;
    ComboMaker.logging  = flag;
    AltSources.logging  = flag;
    ComboSearch.logging = flag;
    //Peco.setLogging(flag);

    LOGGING = flag;
}

//

function log (text) {
    if (LOGGING) {
        console.log(text);
    }
}

//
//

async function main () {
    let needCount;
    let ignoreLoadErrors;

    const opt = CmdLineOptions.parseSystem();
    const options = opt.options;

    // TODO: get rid of this, just pass Opt.options around
    let maxArg = _.toNumber(options.max);
    let useClueList = options.use;
    let showSourcesClueName = options['show-sources'];
    let altSourcesArg = options['alt-sources'];
    let allAltSourcesFlag = options['all-alt-sources'];
    options.allow_dupe_source = options['allow-dupe-source'] ? true : false;
    options.allow_used = options['allow-used'] ? true : false;
    options.merge_style = Boolean(options['merge-style']);
    let showKnownArg = options['show-known'];
    options.copy_from = options['copy-from'];

    if (!maxArg) {
        maxArg = 2; // TODO: default values in opt
    }
    if (!options.count) {
        needCount = true;

        if (showSourcesClueName ||
            altSourcesArg ||
            allAltSourcesFlag ||
            showKnownArg ||
            useClueList ||
            options.test ||
	    options.copy_from
	   ) {
            needCount = false;
        }
        if (needCount) {
            usage('-c COUNT required with those options');
        }
    }

    if (options.output) {
        QUIET = true;
    }

    Validator.setAllowDupeFlags({
        allowDupeNameSrc: false,
        allowDupeSrc:     options.allow_dupe_source,
        allowDupeName:    true
    });

    if (_.includes(options.flags, '2')) {
        ignoreLoadErrors = true;
        Debug('ignoreLoadErrors=true');
    }

    let clueSource = Clues.getByOptions(options);

    setLogging(_.includes(options.verbose, VERBOSE_FLAG_LOAD));
    if (!loadClues(clueSource, ignoreLoadErrors)) {
        return 1;
    }
    setLogging(options.verbose);

    options.notebook = options.notebook || Note.getWorksheetName(clueSource);

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
    } else if (options.test) {
	let start = new Date();
        if (options.validate) {
            Components.validate(options.test, options);
        } else {
            Components.show(options);
        }
	Timing(`count: ${Validator.count}, dupe: ${Validator.dupe}`);
	const d = new Duration(start, new Date()).milliseconds;
	Timing(`${PrettyMs(d)}`);

    } else if (showSourcesClueName) {
        showSources(showSourcesClueName);
    } else if (altSourcesArg || allAltSourcesFlag) {
        AltSources.show(allAltSourcesFlag ? {
            all    : true,
            output : options.output,
            count  : _.toNumber(options.count)
        } : {
            all    : false,
            name   : altSourcesArg,
            output : options.output
        });
    } else if (options.copy_from) {
	const from = Clues.getByVariety(options.copy_from);
	Debug(`from: ${from.baseDir}`);
	copyClues(from, options);
    } else if (options.count) {
        let sources = options['primary-sources'];
        if (options.inverse) {
            sources = ClueManager.getInversePrimarySources(sources.split(',')).join(',');
            console.log(`inverse sources: ${sources}`);
        }
        return combo_maker({
            sum:     options.count,
            max:     maxArg,
            require: options['require-counts'],
            sources,
            use:     useClueList,
            primary: options.primary
        }, options);
    }
}

//

const appBegin = new Date();
main().catch(err => {
    console.error(err, err.stack);
});
if (LOGGING && !QUIET) {
    console.log('runtime: ' + (new Duration(appBegin, new Date())).seconds + ' seconds');
}
