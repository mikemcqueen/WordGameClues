/*
 * make-dict.js
 */

'use strict';

const _ = require('lodash');
const Debug = require('debug')('make-dict');
const Es = require('event-stream');
const Fs = require('fs');
const Opt = require('node-getopt')
      .create([
          ['h', 'help',                'this screen' ]
      ])
      .bindHelp(
          'Usage: node make-dict <text-file>'
      );

const skipped = new Set();
let total_skipped = 0;
let total_duplicates = 0;

//
//
//

function base (word) {
    // strip l' prefix
    const l_apo = 'l\'';
    if (_.startsWith(word, l_apo)) {
        return word.slice(l_apo.length, word.length);
    }

    // strip ' suffix from words with allowed preceding character
    // NOTE: doesn't seem to be working. 
    const apo_suffix_chars = 'ns';
    if (_.endsWith(word, '\'') &&
        _.includes(apo_suffix_chars, word.charAt(word.length - 1))) {

        return word.slice(0, word.length - 1);
    }

    // strip suffixes
    const exts = ['\'s', 'n\'t', '\'ve'];
    for (const ext of exts) {
        if (_.endsWith(word, ext)) {
            return word.slice(0, word.length - ext.length);
        }
    }
    return word;
}

function keep (word) {
    //const allowed = [];
    // keep allowed words
    //if (_.includes(allowed, word)) return true;

    // keep o'prefix
    if (_.startsWith(word, 'o\'') && word.length > 2) return true;

    // don't keep remaining contractions
    if (_.includes(word, '\'')) return false;

    // keep everything else
    return true;
}

function valid (word) {
    if (keep(word)) return true;
    Debug(`skip: ${word}`);
    skipped.add(word);
    total_skipped += 1;
    return false;
}

function clean (word) {
    Debug(`clean in : ${word}`);
    const repl = _.flatMap(word.replace(/[0-9,.:;\"\?\!\(\)_]+/g, ' ').split(' '), x => x)
              .filter(x => !_.isEmpty(x))
              .map(x => base(x))
              .filter(x => valid(x));
    Debug(`clean out: ${repl}`);
    return repl;
}

function get_words (line) {
    let words = _.flatMap(line.split(' '), word => word.split('-'));
    Debug(`words: ${words}, ${words.length}`);
    return _.flatMap(words, word  => clean(_.toLower(word)));
}


function process (filename, set) {
    return new Promise((resolve, reject) => {
        const s = Fs.createReadStream(filename)
            .pipe(Es.split())
            .pipe(Es.mapSync(line => {
                
                // pause the readstream
                s.pause();
                
                Debug(`line: ${line}`);
                const words = get_words(line);
                Debug(`before set: ${_.size(set)}, words: ${_.size(words)}`);
                words.forEach(word => {
		    if (set.has(word)) {
			Debug(`Duplicate: ${word}`);
			total_duplicates += 1;
		    } else {
			set.add(word);
		    }
		});
                Debug(`after set: ${_.size(set)}`);
                
                // resume the readstream, possibly from a callback
                s.resume();
            })).on('error', err => {
                console.error('Error while reading file.', err);
                reject();
            }).on('end', () => {
                Debug('Read entire file.');
                resolve();
            });
    });
}


async function main() {
    const opt = Opt.parseSystem();
    if (opt.argv.length < 1) {
        Opt.showHelp();
        return 1;
    }

    let set = new Set();

    for (const filename of opt.argv) {
        //let filename = opt.argv[0];
        Debug(`filename: ${filename}`);

        await process(filename, set).catch(err => { throw err; });
    }

    console.error('words:');
    set.forEach(word => { console.log(word); });
    console.error(`  skipped unique: ${skipped.size}`);
    console.error(`   skipped total: ${total_skipped}`);
    console.error(`duplicates total: ${total_duplicates}`);
}

main().catch(err =>  {
    console.log(`error, ${err}`);
    console.log(err.stack);
});
