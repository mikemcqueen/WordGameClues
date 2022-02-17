'use strict';

import * as _ from 'lodash';
import * as Words from "../modules/words.js";
const Debug = require('debug')('anfilter');
const Fs    = require('fs-extra');
const Readlines    = require('n-readlines');
const Opt   = require('node-getopt')
      .create([
          ['h', 'help',                'this screen' ]
      ])
      .bindHelp(
          'Usage: node anfilter <ancc output file>'
      );

let old_filter = async (filename: string) => {
    const dict_filename = "words";
    Debug(`dict_filename: ${dict_filename}`);
    
    const dict = await Words.load_dict(dict_filename);
    console.error(`dict(${dict.size})`);
    const result = await Words.load_anresult(filename, dict);
    console.error(`result(${result.length})`);
    
    //result.forEach(line => { console.log(line); } );
    return 0;
};

//
//
let countWordsOfLength = (words: string[], wordLen: number): number => {
    return words.reduce((count: number, word: string) => (word.length === wordLen) ? count + 1 : count, 0);
};

//
//
let new_filter = (filename: string, options: any) => {
    if (options.argv.length < 3) {
        console.log('usage: anfilter <filename> <word-len> <num-words-allowed>');
        return 1;
    }

    let wordLen: number = _.toNumber(options.argv[1]);
    let numWords: number = _.toNumber(options.argv[2]);

    console.log(`len: ${wordLen}, num: ${numWords}`);

    let readLines = new Readlines(filename);
    while (true) {
        let line = readLines.next(); 
        if (line === false) break;
        let count = countWordsOfLength(line.toString().split(' '), wordLen);
        if (count !== numWords) continue;
        console.log(line.toString());
    }
    return 0;
}

let main = async () => {
    const opt = Opt.parseSystem();
    if (opt.argv.length < 1) {
        Opt.showHelp();
        return 1;
    }

    const filename = opt.argv[0];
    Debug(`filename: ${filename}`);

    //return old_filter(filename);
    return new_filter(filename, opt);
}

main().catch(err => {
    console.log(err, err.stack);
});
