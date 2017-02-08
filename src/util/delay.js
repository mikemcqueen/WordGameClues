'use strict';

var _ = require('lodash');

const Seconds = 1;
const Minutes = 2;

var DelayExports = {
    between: between,

    Seconds: Seconds,
    Minutes: Minutes,
}

module.exports = DelayExports;

function between(lo, hi, unit) {
    var value;

    if (unit !== Seconds && unit !== Minutes) {
	throw new Error('invalid unit, ' + unit);
    }
    if (hi < lo) {
	throw new Error('hi < lo, ' + hi + ' < ' + lo);
    }
    
    value = Math.random() * (hi - lo) + lo;
    
    value *= 1000;
    if (unit === Minutes) {
	value *= 60;
    }
    return _.toInteger(value);
}
