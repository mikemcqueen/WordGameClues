//
// test-search-result.js
//

const _            = require('lodash');
const Expect       = require('should/as-function');
const Path         = require('path');
const SearchResult   = require('../search-result');

let wordList = [ 'one', 'two' ];
let filename = 'one-two.json';
let args = { dir: null };
let path = SearchResult.pathFormat({
//          root: args.root,
            dir:  args.dir || _.toString(wordList.length),
            base: filename
        });
console.log(path);
