/*
 * options.js
 */

'use strict';

const _ = require('lodash');

module.exports = exports = new options(); // return a singleton

function options () {
    if (!this.options) {
        this.options = {};
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
