//
// synonym.ts
//

'use strict';

import _ from 'lodash';
import * as Cheerio from 'cheerio';
import Fetch from 'node-fetch';
import * as Synonym from '../dist/modules/synonym';

const Assert       = require('assert');
const My           = require('../../modules/util');
const Opt          = require('node-getopt');
const Stringify    = require('stringify-object');
const Stringify2   = require('javascript-stringify').stringify;

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
    let response;
    let n = 0;
    while (true) {
        response = await Fetch(powerThesaurusUrl(word), options);
        if (response.ok) break;
        console.error(`HTTP: ${response.status}`);
        await My.waitFor(61000);
    }
    const body = await response.text();
    //console.log(body);
    const $ = Cheerio.load(body);
    const at = $('a[title~=synonym]', '#content-list');
    if (at.length === 0) {
        console.log('no results');
        process.exit(-1);
    } else {
        const retryDelay = 31000;
        let synList: Synonym.ListData = { list: [] };
        at.each((i, elem) => {
            const name: string = $(elem).html() || '';
            Assert(!_.isEmpty(name));
            let synData: Synonym.Data = { name, ignore: true };
            synList.list.push(synData);
            //console.log(`${i}: ${$(elem).html()}`);
        });
        console.log(Stringify(synList));
    }
}

main().catch(e => { throw e; });
