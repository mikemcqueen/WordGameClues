//
// clues.js
//

'use strict';

const _           = require('lodash');

const ClueList    = require('../dist/types/clue-list');
const ClueManager = require('../dist/modules/clue-manager');
const ComboMaker  = require('../dist/modules/combo-maker');
const Components  = require('../dist/modules/components');
const MinMax      = require("../dist/types/min-max");
const NameCount   = require('../dist/types/name-count');
const Native      = require('../../build/experiment.node');

const Assert      = require('assert');
const Clues       = require('../modules/clue-types');
const Debug       = require('debug')('clues');
const Duration    = require('duration');
const Expect      = require('should/as-function');
const Log         = require('../modules/log')('clues');
const My          = require('../modules/util');
const Note        = require('../modules/note');
const Opt         = require('node-getopt');
const Peco        = require('../modules/peco');
const PrettyMs    = require('pretty-ms');
const Stringify   = require('stringify-object');
const Timing      = require('debug')('timing');

const CmdLineOptions = Opt.create(_.concat(Clues.Options, [
    ['o', 'output',                            'output json -or- clues(huh?)'],
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
//  ['k', 'show-known',                        'show compatible known clues; -u <clue> required' ],
//  ['',  'csv',                               '  output in search-term csv format' ],
//  ['',  'files',                             '  output in result file full-path format' ],
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
    ['m', 'max-sources=COUNT',                 'enforce COUNT max primary sources for a single clue; default 19\n' +
        '                                            impacts clue loading, combo generation, consistency checking, etc.'],
    ['R', 'remove-all-invalid',                'remove all invalid (validation error) clues'],
    ['',  'ccc',                               'clue (source) consistency check (--save to save results)'],
    ['',  'show-pairs',                        'show unique known source pairs'],
    ['',  'flip',                              '  include flipped (reversed) pairs in results'],
    ['z', 'flags=OPTION+',                     'flags: 2=ignoreErrors,3=cccV2' ],
    ['v', 'verbose',                           'more output' ],
    ['',  'vv',                                'More' ],
    ['',  'vvv',                               'MOAR' ],
    ['',  'memory',                            'output memory dumps' ],
    ['',  'allocations',                       'output memory allocations' ],
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
    ClueManager.loadAllClues({
        addVariations: !!options.test,
        clues,
        ignoreErrors: options.ignoreErrors,
        fast: !options.slow,
        max,
        max_sources: options.max_sources,
        memory: options.memory,
        quiet: options.quiet,
        removeAllInvalid: options.removeAllInvalid,
        useSentences: true,
        validateAll: true,
        verbose: options.verbose
    });
    if (options.memory) {
        Native.dumpMemory("after loadclues");
    }
    return true;
}

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

    const src_lists = Native.getSourceListsForNc(nc);
    if (!src_lists) {
        usage('explosion');
    }
    src_lists.forEach(src_list => {
        src_list.forEach(src => {
            //console.log(Stringify(result.resultMap.map(), { indent: '  ' }));
            console.log(`pnsl:   ${NameCount.listToStringList(src.primaryNameSrcList)}`);
            if (verbose) {
                console.log(`ncList: ${NameCount.listToStringList(src.ncList)}`);
                //result.resultMap.dump();
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
    //Peco.setLogging(flag);

    LOGGING = flag;
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

const show_pairs = (clueSource, max, options) => {
    let pairs = new Set();
    const dir = clueSource.baseDir;
    for (let count = 2; count < max; ++count) {
        let list;
        try {
            list = ClueManager.loadClueList(count, { dir });
        } catch (err) {
            if (!_.includes(err.message, 'ENOENT')) throw err;
            console.error(err);
            continue;
        }
        for (let clue of list) {
            const src = clue.src;
            pairs.add(src);
            if (options.flip) {
                const src_list = src.split(',');
                if (src_list.length !== 2) {
                    console.error(`${src}`);
                } else {
                    const flipped_src = `${src_list[1]},${src_list[0]}`;
                    pairs.add(flipped_src);
                }
            }
        }
    }
    const sorted_pairs = [...pairs].sort();
    for (let pair of sorted_pairs) {
        console.log(pair);
    }
};


async function main () {
    let needCount;
    let ignoreErrors;

    const opt = CmdLineOptions.parseSystem();
    const options = opt.options;

    // default log_level (in c++) is 0. verbose options increase it.
    if (options.vvv) options.verbose = 3;          // ludicrous
    else if (options.vv) options.verbose = 2;      // extra-verbose
    else if (options.verbose) options.verbose = 1; // verbose

    Native.setOptions(options);

    // TODO: get rid of this, just pass Opt.options around
    let useClueList = options.use;
    let showSourcesClueName = options['show-sources'];
    options.allow_dupe_source = options['allow-dupe-source'] ? true : false;
    options.allow_used = options['allow-used'] ? true : false;
    options.merge_style = Boolean(options['merge-style']);
    options.removeAllInvalid = Boolean(options['remove-all-invalid']);
//    let showKnownArg = options['show-known'];
    options.copy_from = options['copy-from'];
    options.max_sources = _.toNumber(options['max-sources'] || 19);
    console.error(`max_sources(${options.max_sources})`);
    options.maxArg = _.toNumber(options.max || 0);  // TODO: make this not used
    let maxArg = _.toNumber(options.max || 2);
    if (!options.count) {
        needCount = true;
        if (showSourcesClueName ||
//            showKnownArg ||
            useClueList ||
            options.test ||
            options.copy_from ||
            options['sort-all-clues'] ||
            options['show-pairs'] ||
            options['ccc'])
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
    if (options['show-pairs']) {
        show_pairs(clueSource, load_max, options);
        process.exit(0);
    }

    if (options.count && /*!showKnownArg && */ !options['ccc']) {
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

    options.notebook = options.notebook || Note.getWorksheetName(clueSource);

    // TODO: add "show.js" with these exports
/*
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
    } else
*/
    if (options.test) {
        let start = new Date();
        if (options.validate) {
            Assert(false);
            //Components.validate(options.test, options);
        } else {
            Components.show(options);
        }
        //Timing(`count: ${Validator.count}, dupe: ${Validator.dupe}`);
        const d = new Duration(start, new Date()).milliseconds;
        Timing(`${PrettyMs(d)}`);
    } else if (showSourcesClueName) {
        showSources(showSourcesClueName);
    } else if (options['ccc']) {
        //if (_.includes(options.flags, '3')) {
        //} else {
        Components.consistency_check(options);
        //}
    } else if (options.copy_from) {
        const from = Clues.getByVariety(options.copy_from);
        Debug(`from: ${from.baseDir}`);
        copyClues(from, options);
    } else if (options.count) {
        // TODO: this is dumb. just pass options.
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
            memory:   options.memory,
            allocations: options.allocations,
            ignoreErrors: options.ignoreErrors,
            max_sources: options.max_sources,
            tpb: options.tpb ? Number(options.tpb) : 0,
            streams: options.streams ? Number(options.streams) : 0,
            stride: options.stride ? Number(options.stride) : 0,
            iters: options.iters ? Number(options.iters) : 0
        });
    }
    return 0;
}

main().catch(err => {
    console.error(err, err.stack);
});
