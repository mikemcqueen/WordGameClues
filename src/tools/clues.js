//
// clues.js
//

'use strict';

const _           = require('lodash');

const ClueList    = require('../dist/types/clue-list');
const ClueManager = require('../dist/modules/clue-manager');
const ComboMaker  = require('../dist/modules/combo-maker');
const Components  = require('../dist/modules/show-components');
const MinMax      = require("../dist/types/min-max");
const NameCount   = require('../dist/types/name-count');
const Validator   = require('../dist/modules/validator');

const AltSources  = require('../modules/alt-sources');
const Clues       = require('../modules/clue-types');
const ComboSearch = require('../modules/combo-search');

const Debug       = require('debug')('clues');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Log         = require('../modules/log')('clues');
const Note        = require('../modules/note');
const Opt         = require('node-getopt');
const Peco        = require('../modules/peco');
const PrettyMs    = require('pretty-ms');
const Show        = require('../modules/show');
const Stringify   = require('stringify-object');
const ResultMap   = require('../types/result-map');
const Timing      = require('debug')('timing');


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
//  ['',  'and=NAME[:COUNT][,NAME[:COUNT]]+',  '  combos must have source NAME[:COUNT]'],
    ['',  'xor=NAME[:COUNT][,NAME[:COUNT]]+',  '  combos must not have, and must be compatible with, source NAME[:COUNT]s'],
    ['',  'or=NAME[:COUNT][,NAME[:COUNT]]+',   '  combos must either have, or be compatible with, source NAME[:COUNT]s'],
    ['',  'primary',                           '  show combos as primary source clues' ],
    ['l', 'parallel',                          '  use paralelljs' ],
    ['',  'slow',                              '  use (old) slow method of loading clues' ],
    ['',  'streams=COUNT',                     '  use COUNT streams (CUDA)'],
    ['',  'stride=COUNT',                      '  use COUNT source indices per stream (CUDA)'],
    ['',  'tpb=COUNT',                         '  use COUNT threads per block (CUDA)'],
    ['',  'iters=COUNT',                       '  run COUNT iterations (only valid if no range specified with -c)'],
    ['',  'copy-from=SOURCE',                  'copy clues from source cluetype; e.g. p1.1'],
    ['',  'save',                              '  save clue files'],
    ['',  'allow-dupe-source',                 '  allow duplicate sources'],
    ['',  'merge-style',                       '  merge-style, no validation except for immediate sources'],
    ['',  'remaining',                         '  only word combos not present in any named note'],
    ['i', 'primary-sources=SOURCE[,SOURCE,...]','limit results to the specified primary SOURCE(s)'],
    ['',  'inverse',                           '  or the inverse of those source(s); use with -i'],
    ['k', 'show-known',                        'show compatible known clues; -u <clue> required' ],
    ['',  'csv',                               '  output in search-term csv format' ],
    ['',  'files',                             '  output in result file full-path format' ],
    ['s', 'show-sources=NAME[:COUNT][,v]',     'show primary source combos for the specified NAME[:COUNT]' ],
    ['t', 'test=SOURCE[,SOURCE,...]',          'test the specified source list, e.g. blue,fish' ],
    ['',  'add=NAME',                          '  add compound clue NAME=SOURCE; use with --test' ],
    ['',  'remove=NAME',                       '  remove compound clue NAME=SOURCE; use with --test' ],
    ['',  'property=PROP',                     '    add PROP:true, or remove PROP from existing compound clues' +
                                                  ' matching NAME=SOURCE; use with --test --add/--remove' ],
    ['',  'reject',                            '  add combination to reject list; use with --test' ],
    ['',  'fast',                              '  use fast method' ],
    ['',  'validate',                          '  treat SOURCE as filename, validate all source lists in file'],
    ['',  'combos',                            '    validate all combos of sources/source lists in file'],
    ['u', 'use=NAME[:COUNT]+',                 'use the specified NAME[:COUNT](s)' ],
    ['',  'allow-used',                        '  allow used clues in clue combo generation' ],
    ['',  'any',                               '  any match (uh, probably should not use this)'],
    ['',  'production',                        'use production note store'],
    ['',  'sort-all-clues',                    'sort all clue data files by src'],
    ['m', 'max-sources=COUNT',                 'enforce COUUNT max primary sources for a single clue;' +
                                               ' impacts clue loading, combo generation'],
    ['R', 'remove-all-invalid',                'remove all invalid (validation error) clues'],
    ['z', 'flags=OPTION+',                     'flags: 2=ignoreErrors' ],
    ['v', 'verbose',                           'more output'],
    ['q', 'quiet',                             'less output'],
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

const loadClues = (clues, max, options) => {
    log('loading all clues...');
    ClueManager.loadAllClues({
        clues,
        max,
        max_sources: options.max_sources,
        useSentences: true,
        quiet: options.quiet,
        ignoreErrors: options.ignoreErrors,
        fast: !options.slow,
        addVariations: !!options.test,
        validateAll: true,
        removeAllInvalid: options.removeAllInvalid
    });
    log('done.');
    return true;
}

/*
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
    // is _chain even necessary here?
        args.require = _.chain(args.require).split(',').map(_.toNumber).value();
    }
}
    */

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

function combo_maker(args) {
    return Promise.resolve(args.remaining ? getNamedNoteNames(args) : false)
        .then(noteNames => {
            if (noteNames) {
                Log.info(`note names: ${noteNames}`);
                args.note_names = noteNames;
            }
            return ComboMaker.makeCombos(args);
        });
}

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

function copyClues (fromType, options = {}) {
    const dir = fromType.baseDir;
    let total = 0;
    let copied = 0;
    for (let count = 2; count < options.max_sources; ++count) {
        let list;
        try {
            list = ClueManager.loadClueList(count, { dir });
        } catch (err) {
            if (!_.includes(err.message, 'ENOENT')) throw err;
            console.error(err);
            continue;
        }
        total += _.size(list);
        //Debug(`count ${count}: size(${_.size(list)})`);
        console.log(`count(${count}):\n${Stringify(list)}`);
        for (let clue of list) {
            const nameList = clue.src.split(',').sort();
            let countSet;
            let compound = false;
            if (0) { // fast, but no respeto restrictToSameClueNumberOnly flag
                const ncLists = ClueManager.fast_getCountListArrays(clue.src, { add: true, fast: true });
                if (_.isEmpty(ncLists)) {
                    Debug(`No matches for: ${clue.src}`);
                    continue;
                }
                countSet = ncLists.reduce((countSet, ncList) => {
                    let sum = ncList.reduce((sum, nc) => { return sum + nc.count; }, 0);
                    if (sum <= max) countSet.add(sum);
                    return countSet;
                }, new Set());
            } else { // slower, with respect
                const result = ClueManager.getCountListArrays(clue.src, { add: true, fast: true });
                if (!result) {
                    Debug(`No matches for: ${clue.src}`);
                    continue;
                }
                //{ valid, known, rejects, invalid, clues, addRemoveSet };
                console.log(` valid(${_.size(result.valid)}) known(${_.size(result.known)}) invalid(${_.size(result.invalid)}) clues(${_.size(result.clues)})`);
                if (_.isEmpty(result.known) && _.isEmpty(result.valid)) continue;
                countSet = new Set();
                countSet = result.valid.reduce((countSet, countList) => {
                    let sum = countList.reduce((sum, count) => sum + count, 0);
                    if (sum <= max) countSet.add(sum);
                    // only add for the current count
                    //if (sum === count) countSet.add(sum);
                    return countSet;
                }, countSet);
                countSet = result.known.reduce((countSet, obj) => {
                    let sum = obj.countList.reduce((sum, count) => sum + count, 0);
                    if (sum <= max) countSet.add(sum);
                    // only add for the current count
                    //if (sum === count) countSet.add(sum);
                    return countSet;
                }, countSet);
                compound = true;
            }
            
            Debug(`Adding ${clue.name}:${clue.src}, counts: ${Array.from(countSet)}`);
            const added = ClueManager.addRemoveOrReject({
                add: clue.name,
                isReject: false,
            }, nameList, countSet, { save: options.save, compound });
            copied += added;
        }
    }
    console.log(`total: ${total}, copied: ${copied}`);
}

function setLogging (flag) {
    ClueManager.setLogging(flag);
    ComboMaker.logging  = flag;
    AltSources.logging  = flag;
    ComboSearch.logging = flag;
    //Peco.setLogging(flag);

    LOGGING = flag;
}

function log (text) {
    if (LOGGING) {
        console.log(text);
    }
}

function sortAllClues (clueSource, max) {
    const dir = clueSource.baseDir;
    //max = 3;
    for (let count = 2; count < max; ++count) {
        let list;
        try {
            list = ClueManager.loadClueList(count, { dir });
            if (!_.isEmpty(list)) {
                list.sort((a, b) => a.src.localeCompare(b.src));
                ClueManager.saveClueList(list, count, { dir });
            }
        } catch (err) {
            if (!_.includes(err.message, 'ENOENT')) throw err;
            console.error(err);
            continue;
        }
    }
}

async function main () {
    let needCount;
    let ignoreErrors;

    const opt = CmdLineOptions.parseSystem();
    const options = opt.options;

    // TODO: get rid of this, just pass Opt.options around
    let useClueList = options.use;
    let showSourcesClueName = options['show-sources'];
    let altSourcesArg = options['alt-sources'];
    let allAltSourcesFlag = options['all-alt-sources'];
    options.allow_dupe_source = options['allow-dupe-source'] ? true : false;
    options.allow_used = options['allow-used'] ? true : false;
    options.merge_style = Boolean(options['merge-style']);
    options.removeAllInvalid = Boolean(options['remove-all-invalid']);
    let showKnownArg = options['show-known'];
    options.copy_from = options['copy-from'];
    console.error(`max-sources(${options['max-sources']})`);
    options.max_sources = _.toNumber(options['max-sources'] || 20);
    options.maxArg = _.toNumber(options.max || 0);  // TODO: make this not used
    let maxArg = _.toNumber(options.max || 2);
    if (!options.count) {
        needCount = true;
        if (showSourcesClueName ||
            altSourcesArg ||
            allAltSourcesFlag ||
            showKnownArg ||
            useClueList ||
            options.test ||
            options.copy_from ||
            options['sort-all-clues'])
        {
            needCount = false;
        }
        if (needCount) {
            usage('-c COUNT required with those options');
        }
    }

    if (options.output) {
        QUIET = true;
    }

    /****
    OldValidator.setAllowDupeFlags({
        allowDupeNameSrc: false,
        allowDupeSrc:     options.allow_dupe_source,
        allowDupeName:    true
    });
    ****/

    if (_.includes(options.flags, '2')) {
        options.ignoreErrors = true;
        Debug('ignoreErrors=true');
    }

    let clueSource = Clues.getByOptions(options);
    let load_max = options.max_sources;

    if (options['sort-all-clues']) {
        sortAllClues(clueSource, load_max);
        process.exit(0);
    }

    if (options.count && !showKnownArg && !altSourcesArg && !allAltSourcesFlag) {
        let countRange = options.count.split(',').map(_.toNumber);
        options.count_lo = countRange[0];
        options.count_hi = countRange.length > 1 ? countRange[1] : countRange[0];
        load_max = options.count_hi;
    }

    setLogging(_.includes(options.verbose, VERBOSE_FLAG_LOAD));
    let loadBegin = new Date();
    if (!loadClues(clueSource, load_max, options)) {
        return 1;
    }
    let loadMillis = new Duration(loadBegin, new Date()).milliseconds;
    console.error(`loadClues(${PrettyMs(loadMillis)})`);
    setLogging(false); //options.verbose);

    console.error(`merge_nclc: ${Validator.merge_nclc}`);
//    process.exit(0);

    options.notebook = options.notebook || Note.getWorksheetName(clueSource);

    log(`count=${options.count}, max=${maxArg}`);

    // TODO: add "show.js" with these exports
    if (showKnownArg) {
        if (!useClueList) {
            // TODO: require max if !metaFlag
            console.log('one or more -u NAME:COUNT required with that option');
            return 1;
        }
        Show.compatibleKnownClues({
            nameList: useClueList,
            max: options.count ? Number(options.count) : ClueManager.getMaxClues(),
            root: '../data/results/',
            format: {
                csv: options.csv,
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
            count  : Number(options.count)
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
        /*
        let sources = options['primary-sources'];
        if (options.inverse) {
            sources = ClueManager.getInversePrimarySources(sources.split(',')).join(',');
            console.log(`inverse sources: ${sources}`);
        }
        */
        return combo_maker({
            sum:     options.count,
            max:     maxArg,
            use:     useClueList,
            primary: options.primary,
            apple:   options.apple,
            final:   options.final,
            or:      options.or,
            xor:     options.xor,
            //and:     options.and,
            parallel: options.parallel,
            verbose:  options.verbose,
            quiet:    options.quiet,
            max_sources: options.max_sources,
            tpb: options.tpb ? Number(options.tpb) : 0,
            streams: options.streams ? Number(options.streams) : 0,
            stride: options.stride ? Number(options.stride) : 0,
            iters: options.iters ? Number(options.iters) : 0
        });
    }
    return 0;
}

//

const appBegin = new Date();
main().catch(err => {
    console.error(err, err.stack);
});
if (LOGGING && !QUIET) {
    console.error('runtime: ' + (new Duration(appBegin, new Date())).seconds + ' seconds');
}
