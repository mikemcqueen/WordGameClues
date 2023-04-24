//
// sentence.js
//

'use strict';

//import _ from 'lodash';
//const Clues = require('../../modules/clue-types');
//const Fs = require('fs-extra');
//const Path = require('path');

const Clues = require('../modules/clue-types');
const Sentence = require('../dist/types/sentence');
const Stringify = require('stringify-object');

const main = async () => {
    let variations = Sentence.emptyVariations();
    let options = {
	"apple": "f.71"
    };
    let sentence = Sentence.load(Clues.getDirectory(Clues.getByOptions(options)), 3);
    Sentence.addVariations(sentence, variations);
    const candi = Sentence.buildAllCandidates(sentence, variations);
    console.log(Stringify(candi));
};

main().catch(err => {
    console.error(err, err.stack);
});
