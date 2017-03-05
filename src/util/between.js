// BETWEEN.JS

'use strict';

//

const Expect = require('chai').expect;

//

module.exports = exports = function between(lo, hi) {
    Expect(lo, 'lo').to.be.a('number');
    Expect(hi, 'hi').to.be.a('number');
    Expect(hi, 'hi < lo').to.be.at.least(lo);
    return lo + Math.random() * (hi - lo);
}
