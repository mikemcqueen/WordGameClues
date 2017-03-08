const Delay    = require('../delay');
const ms       = require('milliseconds');
const expect   = require('chai').expect;
const prettyMs = require('pretty-ms');

describe('test delay', function() {

    it('loop 2-3 minute delay', function() {
	let lo = 2, hi = 3;
	for (let count = 0; count < 20; ++count) {
	    let delay = Delay.between(lo, hi, Delay.Minutes);
	    expect(delay).to.be.at.least(ms.minutes(lo)).and.at.most(ms.minutes(hi));
	    console.log(prettyMs(delay));
	}
    });

});
