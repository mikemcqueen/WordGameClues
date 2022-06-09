//
// synonym.ts
//

'use strict';

import _ from 'lodash';

export interface Data {
    name: string;
    active?: boolean;
    ignore?: boolean;
}

export interface ListData {
    list: Data[];
    ignore?: boolean;
}
