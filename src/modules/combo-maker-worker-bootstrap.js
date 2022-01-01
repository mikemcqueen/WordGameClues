/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ComboWorker = require('./combo-maker-worker');
const Stringify   = require('stringify-object');

let bootstrap = (data) => {
    console.log(`bootstrapping`);  //${Stringify(data)}`);
    ComboWorker.doWork(data);
    return 0;
};

let entrypoint = (data) => {
    return bootstrap(data);
};

process.once('message', code => {
    let d = JSON.parse(code).data;
    //console.log(d);
    eval(d);
});

module.exports = {
    entrypoint
};
