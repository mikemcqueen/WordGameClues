// 

const DIR = `${__dirname}/test-files/`;

//

function file (filename) {
    return `${DIR}${filename}`;
}

//

module.exports = {
    DIR,
    file
};
