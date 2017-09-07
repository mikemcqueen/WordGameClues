//
// test-search-result.js
//

const _            = require('lodash');
const Expect       = require('should/as-function');
//const Fs           = require('fs');
//const Google       = require('google');
//const Ms           = require('ms');
//const My           = require('../misc/util');
const Path         = require('path');
//const PrettyMs     = require('pretty-ms');
//const Promise      = require('bluebird');
//const Score        = require('./score');
const SearchResult   = require('../search-result');

let wordList = [ 'one', 'two' ];
let filename = 'one-two.json';
let args = { dir: null };
let path = SearchResult.pathFormat({
//	    root: args.root,
	    dir:  args.dir || _.toString(wordList.length),
	    base: filename
	});
console.log(path);
