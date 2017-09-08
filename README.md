This code was written in the Jan/Jeb 2017 timeframe, and is a good example of my JavaScript around that time.

This repo includes the tools for helping me select clues for a word game I'm designing. It's similar to a crossword puzzle, except that instead of intersecting clue solutions (words) on a grid, the words are themselves used as clues for more deeply nested puzzles.

For example, given clue words:

"blue", "origin" : the solution might be "Jeff Bezos."

"noble", "barn"  : the solution might be "Bookseller."

then,

"Jeff Bezos", "bookseller" : the solution might be "Amazon."

And so on, as deep as you like. The tools I've included in this repo are related to Google/Wikipedia searching for results of word pairs, as above, and managing the results. They consist of:

search.js - search Google for results, given a CSV file of word pairs (or any number of words).

score.js  - score the results generated from Search, based on the frequency/location of source words in a Wikipedia article.

filter.js - output results that meet certain scoring criteria.

Note that search is throttled to send out search requests very slowly (~150/day), because it's not technically following the Google TOS. I also updated search so that it automatically scores, and saves the results so I no longer need to manually run score (unless I want to re-score results).

to install:

npm install

to test:

npm test

to run:

cd src/tools

node search sample.csv 1 2    # search 3 two-word combinations with 1 - 2 minute delay between searches.

node filter -d2               # filter two-word results, showing all URLs that match both words in title.

node filter -d2 -c            # filter two-word results, showing count of above.

node filter -d2 -a            # filter two-word results, showing all URLs that match both words in article.
