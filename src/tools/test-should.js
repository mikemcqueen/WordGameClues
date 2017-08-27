const Expect = require('should/as-function');

let foo = [ 1 ];
let bar = [];

Expect(foo).is.a.Array().which.is.not.empty();
Expect(foo).is.a.Array().and.not.empty();
Expect(bar).is.a.Array().and.empty();
