/*
 * options.js
 */

'use strict';

const _ = require('lodash');

exports = module.exports = new options(); // return a singleton

function options () {
    const self = this;

    if (!this.options) {
	this.options = {};
    }

    this.set = function (options = {}) {
	_.forOwn(options, (value, key) => {
	    //console.log(`adding option: ${key} : ${value}`);
	    self.options[key] = value;
	});
	return self.options;
    };

    return self;
}
