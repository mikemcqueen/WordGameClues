/*
 * options.js
 */

'use strict';

const _ = require('lodash');

module.exports = exports = new options(); // return a singleton

function options (a) {
    if (!this.options) {
	console.log(`options INIT: ${a}`);
	this.options = {};
    } else {
	console.log(`options ${a}`);
    }

    const self = this;
    this.set = function (options = {}) {
	_.forOwn(options, (value, key) => {
	    //console.log(`adding option: ${key} : ${value}`);
	    self.options[key] = value;
	});
	return self.options;
    };

    return this;
}
