console.log(process.argv[2]);
const MidiParse = require("midiconvert");
const fs = require('fs');
var dir = "./" + process.argv[2];
var dirJson = dir + "JSON"
if (!fs.existsSync(dirJson)){
    fs.mkdirSync(dirJson);
}
// const fs = require('fs');
const MidiConvert = require('midiconvert');

fs.readdirSync(`${dir}`).forEach(file => {
    let midiBlob = fs.readFileSync(`${dir}/${file}`, "binary");
    let json = MidiConvert.parse(midiBlob);
    fs.writeFileSync(`${dirJson}/${file}.json`, JSON.stringify(json, null, 4));
});


