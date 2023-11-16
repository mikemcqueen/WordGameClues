//
// consistency.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Native      = require('../../../build/experiment.node');

const Assert      = require('assert');
const Debug       = require('debug')('consistency');
const Duration    = require('duration');
const PrettyMs    = require('pretty-ms');
//const stringify   = require('javascript-stringify').stringify;
//const Stringify2  = require('stringify-object');

//import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
//import * as NameCount from '../types/name-count';
import * as PreCompute from './cm-precompute';
//import * as Sentence from '../types/sentence';
//import * as Source from './source';
