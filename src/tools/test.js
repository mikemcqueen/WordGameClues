const _ = require('lodash');

let test = (x) => { return !(x % 2) ? "yes" : 0; };

for (let i = 0; i < 10; i++) {
    console.log(`${i}: ${test(i)}`);
}

