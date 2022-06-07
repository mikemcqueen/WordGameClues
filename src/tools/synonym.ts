//
//
//

'use strict';

import _ from 'lodash';
import fetch from 'node-fetch';
import * as cheerio from 'cheerio';

const Assert    = require('assert');
const Opt       = require('node-getopt');

const CmdLineOptions = Opt.create([
    ['h', 'help',                              'this screen']
]).bindHelp();

//
//
let powerThesaurusUrl = (word: string): string => {
    return `https://www.powerthesaurus.org/${word}/synonyms`;
}

//
//
let main = async (): Promise<void> => {
    const headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36'
    };
    const options = { headers };

    const opt = CmdLineOptions.parseSystem();
    if (opt.argv.length !== 1) {
        console.log('usage: node synonym <word>');
        process.exit(-1);
    }
    const word = opt.argv[0];
    const response = await fetch(powerThesaurusUrl(word), options);
    Assert(response.ok, `HTTP: ${response.status}`);
    const body = await response.text();
    //console.log(body);
    const $ = cheerio.load(body);
    const at = $('a[title~=synonym]', '#content-list');
    //console.log(`at.length(${at.length}), at[0]:  ${at[0]}`);
    //console.log(`atitles (${atitles.length}) type: ${typeof atitles}`);
    at.each((i, elem) => {
        console.log(`${i}: ${$(elem).html()}`);
    });
}

main().catch(e => { throw e; });
