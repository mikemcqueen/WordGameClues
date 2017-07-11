//
// TEST-UTIL.JS
//

const _            = require('lodash');
const Expect       = require('chai').expect;
const Fs           = require('fs');
const Ms           = require('ms');
const My           = require('../util');
const PrettyMs     = require('pretty-ms');
const Promise      = require('bluebird');

//const fsReadFile   = Promise.promisify(Fs.readFile);
const fsWriteFile  = Promise.promisify(Fs.writeFile);

//

function randomObject() {
    return { a : _.random(), b: _.random() };
}

//

function writeAddCommit(filepath, obj) {
    console.log(`writeAddCommit ${filepath}`);
    console.log('committing file');
    return My.gitCommit(filepath, 'adding test file');
}

//
	  
describe ('delay tests:', function () {

    it.skip ('between: verify 20x 2-3 minute values', function () {
	let lo = Ms('2m');
	let hi = Ms('3m');
	for (let count = 0; count < 20; ++count) {
	    let delay = My.between(lo, hi);
	    Expect(delay).to.be.at.least(lo).and.at.most(hi);
	    console.log(PrettyMs(delay));
	}
    });

});
