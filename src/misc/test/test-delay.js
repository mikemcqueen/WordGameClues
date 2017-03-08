//
// TEST-UTIL.JS
//

const Expect   = require('chai').expect;
const Ms       = require('ms');
const My       = require('../util');
const PrettyMs = require('pretty-ms');

describe ('delay tests:', function() {

    it ('between: verify 20x 2-3 minute values', function() {
	let lo = Ms('2m');
	let hi = Ms('3m');
	for (let count = 0; count < 20; ++count) {
	    let delay = My.between(lo, hi);
	    Expect(delay).to.be.at.least(lo).and.at.most(hi);
	    console.log(PrettyMs(delay));
	}
    });

});
