const express = require('express');
const app = express();
const port = 3000;

var AWS = require('aws-sdk');
AWS.config.update({
    // Put in your access key ID
    accessKeyId: "AKIAZSSVVP446WY2R7GN",
    // Put in your secret access key
    secretAccessKey: "TvPC1VtfEdNOWqvTK5qfcgFNfwqrOW4kei68W4Yj",
    region: 'eu-west-1'
});
const TABLE = 'Movies';
const BUCKET = "csu44000assignment220";
const OBJECT = "moviedata.json";
const INSERT_SET_SIZE = 25;
var s3 = new AWS.S3();
var ddb = new AWS.DynamoDB();

async function sendClientFile(_, res) {
    res.sendFile(__dirname + "/" + "client.html");
}

async function create(_, res) {
    var exists = await checkTableExists();
    if(exists) {
        console.log("Exists");
        returnMessage(res, "success", {});
        return;
    }

    console.log("Creating Table...");
    var json = await getS3Object(BUCKET, OBJECT);

    await createTable();
    await ddb.waitFor('tableExists', { TableName: TABLE}).promise();
    await insertIntoTable(json);
    console.log('Table Created');
    returnMessage(res, "success", {});
    return;
}

async function query(req, res){
    var movie = req.get('Movie');
    var year = parseInt(req.get('Year'));
    var rating = parseInt(req.get('Rating'));

    if(isNaN(year)) {
        returnMessage(res, "Failed, year not set", {});
    } else if(isNaN(rating)) {
        returnMessage(res, "Failed, rating not set", {})
    } else if(movie.length == 0) {
        returnMessage(res, "Failed, no starting text given", {})
    } else {
        console.log('Querying...');
        let data = await queryTable(movie.toLowerCase(), year.toString(), rating.toString());
        console.log('Query Successful');
        returnMessage(res, "Success", data);
    }
}

async function destroy(_, res) {
    var exists = await checkTableExists();
    if(!exists) {
        console.log("Doesn't Exists");
        returnMessage(res, "success", {});
        return;
    }

    console.log("Deleting Table...");
    var params = { TableName: TABLE };
    await ddb.deleteTable(params).promise();
    console.log("Table Deleted");
    returnMessage(res, "success", {});
}

async function checkTableExists() {
    var allTables = await ddb.listTables({}).promise();
    return allTables.TableNames.includes(TABLE);
}

async function getS3Object() {
    var params = {
        Bucket: BUCKET,
        Key: OBJECT
    };
    var data = await s3.getObject(params).promise();
    return JSON.parse(data.Body.toString('utf-8'));
}

async function createTable() {
    var params = {
        TableName: TABLE,
        KeySchema: [
            { AttributeName: 'title', KeyType: 'HASH' },
            { AttributeName: 'release_date', KeyType: 'RANGE'}
        ],
        AttributeDefinitions: [
            { AttributeName: 'title', AttributeType: 'S' },
            { AttributeName: 'release_date', AttributeType: 'N' }
        ],
        ProvisionedThroughput: {
            ReadCapacityUnits: 1,
            WriteCapacityUnits: 1
        }
    }
    await ddb.createTable(params).promise();
}

async function insertIntoTable(json) {
    var sets = [], movies = [];
    for(var i = 0; i < json.length; i++) {
        if(movies.length == INSERT_SET_SIZE) {
            sets.push(movies);
            movies = [];
        }

        title = json[i].title;
        lowerTitle = json[i].title.toLowerCase();
        if(json[i].hasOwnProperty('year')) year = json[i].year.toString();
        else year = '-1';
        if(json[i].info.hasOwnProperty('rating')) rating = json[i].info.rating.toString();
        else rating = '-1';
        if(json[i].info.hasOwnProperty('rank')) rank = json[i].info.rank.toString();
        else rank = '-1';

        movies.push({
            PutRequest: {
                Item: {
                    title: { 'S': title },
                    release_date: { 'N': year },
                    rating: { 'N': rating },
                    lowerCaseTitle: {'S': lowerTitle},
                    rank: { 'N': rank }
                }
            }
        });
    }
    if(movies.length != 0) sets.push(movies);

    for(var i = 0; i < sets.length; i++) {
        var percentComplete = (((i+1) / (sets.length+1)) * 100).toFixed(2);
        process.stdout.write(`\rData insertion progress: ${percentComplete}% complete\r`);
        await ddb.batchWriteItem({ RequestItems: { [TABLE]: sets[i] } }).promise();
    }
    process.stdout.write(`\rData insertion progress: 100% complete\r\n`)
}

async function queryTable(movie, year, rating) {
    var params = {
        TableName: TABLE,
        ExpressionAttributeValues: {
            ':y': {N: year},
            ':t': {S: movie},
            ':r': {N: rating}
        },
        FilterExpression: 'release_date = :y and begins_with (lowerCaseTitle, :t) and rating >= :r'
    }

    var res = await ddb.scan(params).promise();
    var data = [];

    res.Items.forEach(function(item, _, _) {
        data.push({
            title: item.title.S,
            year: item.release_date.N,
            rating: item.rating.N,
            rank: item.rank.N
        });
    });

    return data;
}

function returnMessage(res, message, data) {
    body = {
        message: message,
        data: data
    };
    res.send(body);
    return;
}

app.use(express.json());
app.use(express.static('public'));
app.get('/', sendClientFile);
app.get('/create', create);
app.get('/query', query);
app.get('/destroy', destroy);
app.listen(port, () => console.log(`Example app listening on port ${port}!`));