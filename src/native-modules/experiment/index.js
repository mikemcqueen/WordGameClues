const xp = require('./build/Release/experiment.node');

let greet = () => {
    console.log('exports: ', xp);
    console.log();
    console.log('xp.greetHello(): ', xp.greetHello());
    console.log();
};

const mapEntries = [
[
 'us 50:7',
 [
  {
   primaryNameSrcList: [
    {
     name: 'ace',
     count: 32,
     index: undefined
    },
    {
     name: 'f',
     count: 31,
     index: undefined
    },
    {
     name: 'lily',
     count: 56,
     index: undefined
    },
    {
     name: 'white',
     count: 48,
     index: undefined
    },
    {
     name: 'man o war',
     count: 3,
     index: undefined
    },
    {
     name: 'buffalo spring',
     count: 1,
     index: undefined
    },
    {
     name: 'canyon',
     count: 4,
     index: undefined
    }
   ],
   sourceNcCsvList: [
    'constitution:5,street:2',
    'constitution:5',
    'american:4,man o war:1',
    'american:4',
    'cafe:2,casablanca:2',
    'cafe:2',
    'ace:1,f:1',
    'ace:1',
    'f:1',
    'casablanca:2',
    'lily:1,white:1',
    'lily:1',
    'white:1',
    'man o war:1',
    'street:2',
    'buffalo spring:1,canyon:1',
    'buffalo spring:1',
    'canyon:1',
    'us 50:7'
   ],
   ncList: [
    {
     name: 'us 50',
     count: 7
    }
   ],
   synonymCounts: {
    total: 1,
    primary: 0
   }
  }
 ]
],
[
 '21:8',
 [
  {
   primaryNameSrcList: [
    {
     name: 'buffalo',
     count: 71,
     index: undefined
    },
    {
     name: 'soldier',
     count: 3,
     index: undefined
    },
    {
     name: 'bird',
     count: 50,
     index: undefined
    },
    {
     name: 'red',
     count: 29,
     index: undefined
    },
    {
     name: 'star smith',
     count: 2,
     index: undefined
    },
    {
     name: 'lily',
     count: 56,
     index: undefined
    },
    {
     name: 'tiger',
     count: 55,
     index: undefined
    },
    {
     name: 'volley',
     count: 54,
     index: undefined
    }
   ],
   sourceNcCsvList: [
    'buffalo soldier:2,san juan hill:6',
    'buffalo soldier:2',
    'buffalo:1,soldier:1',
    'buffalo:1',
    'soldier:1',
    'san juan hill:6',
    'hill:2,san juan:4',
    'hill:2',
    'bird:1,red:1',
    'bird:1',
    'red:1',
    'san juan:4',
    'star smith:1,volleyball:3',
    'star smith:1',
    'volleyball:3',
    'ball:2,volley:1',
    'ball:2',
    'lily:1,tiger:1',
    'lily:1',
    'tiger:1',
    'volley:1',
    '21:8'
   ],
   ncList: [
    {
     name: '21',
     count: 8
    }
   ],
   synonymCounts: {
    total: 3,
    primary: 1
   }
  }
 ]
]

];

const ncDataLists = [
[
 {
  ncList: [
   {
    name: 'us 50',
    count: 7
   }
  ]
 }
]
/*
,
 {
  ncList: [
   {
    name: '21',
    count: 8
   }
  ]
 }
],
[
 {
  ncList: [
   {
    name: 'us 50',
    count: 7
   }
  ]
 },
 {
  ncList: [
   {
    name: '21',
    count: 8
   }
  ]
 }
]
*/

/*
,
 {
  ncList: [
   {
    name: 'koa',
    count: 2
   }
  ]
 },
 {
  ncList: [
   {
    name: 'currant',
    count: 4
   }
  ]
 }
],
[
 {
  ncList: [
   {
    name: 'us 50',
    count: 7
   }
  ]
 },
 {
  ncList: [
   {
    name: '21',
    count: 8
   }
  ]
 },
 {
  ncList: [
   {
    name: 'koa',
    count: 2
   }
  ]
 },
 {
  ncList: [
   {
    name: 'currant',
    count: 5
   }
  ]
 }
],
[
 {
  ncList: [
   {
    name: 'us 50',
    count: 7
   }
  ]
 },
 {
  ncList: [
   {
    name: '21',
    count: 8
   }
  ]
 },
 {
  ncList: [
   {
    name: 'koa',
    count: 2
   }
  ]
 },
 {
  ncList: [
   {
    name: 'currant',
    count: 6
   }
  ]
 }
]
*/
    
];

let buildSourceListsForUseNcData = () => {
    xp.buildSourceListsForUseNcData(ncDataLists, mapEntries, undefined);
};

let mergeCompatibleXorSourceCombinations = () => {
    xp.mergeCompatibleXorSourceCombinations(ncDataLists, mapEntries);
};

greet();
buildSourceListsForUseNcData();
mergeCompatibleXorSourceCombinations();
