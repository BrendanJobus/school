const express = require('express');
const app = express();
var AWS = require('aws-sdk');
AWS.config.update({
    accessKeyId: "AKIAZSSVVP446WY2R7GN",
    secretAccessKey: "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",
    region: 'eu-west-1'
});
const port = 3000;

app.use(express.json());
app.use(express.static('public'));
app.get('/', sendClientFile);
app.get('/create', createTable);
app.get('/query', queryTable);
app.get('/destroy', destroyTable);

const BUCKET = "csu44000assignment220";
const OBJECT = "moviedata.json";
const TABLE = 'Movies'
const BATCH_SIZE = 25;

var s3 = new AWS.S3();
var ddb = new AWS.DynamoDB();

async function sendClientFile(_, res) {
    res.sendFile(__dirname + "/" + "client.html");
}

async function createTable(_, res) {
    var exists = await checkTableExists();
    if(exists) {
        console.log("Exists");
        returnMessage(res, "success", {});
        return;
    }

    console.log("Creating...");
    var json = await getS3Object(BUCKET, OBJECT);

    await createDynamoTable();
    await ddb.waitFor('tableExists', { TableName: TABLE}).promise();
    await insertIntoDynamoTable(json);
    console.log('Done!');
    returnMessage(res, "success", {});
    return;
}

async function queryTable(req, res){
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
        let data = await queryDynamoTable(movie.toLowerCase(), year.toString(), rating.toString());
        console.log('Done!');
        returnMessage(res, "Success", data);
    }
}

async function destroyTable(_, res) {
    var exists = await checkTableExists();
    if(!exists) {
        console.log("Doesn't Exists");
        returnMessage(res, "success", {});
        return;
    }

    console.log("Deleting...");
    await destroyDynamoTable();
    console.log("Done!");
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

async function createDynamoTable() {
    var params = {
        AttributeDefinitions: [
            { AttributeName: 'title', AttributeType: 'S' },
            { AttributeName: 'release_date', AttributeType: 'N' }
        ],
        KeySchema: [
            { AttributeName: 'title', KeyType: 'HASH' },
            { AttributeName: 'release_date', KeyType: 'RANGE'}
        ],
        ProvisionedThroughput: {
            ReadCapacityUnits: 1,
            WriteCapacityUnits: 1
        },
        TableName: TABLE
    }
    await ddb.createTable(params).promise();
}

async function insertIntoDynamoTable(json) {
    var batches = [], batch = [];
    for(var i = 0; i < json.length; i++) {
        if(batch.length == BATCH_SIZE) {
            batches.push(batch);
            batch = [];
        }

        title = json[i].title;
        lowerTitle = json[i].title.toLowerCase();
        if(json[i].hasOwnProperty('year')) year = json[i].year.toString();
        else year = '-1';
        if(json[i].info.hasOwnProperty('rating')) rating = json[i].info.rating.toString();
        else rating = '-1';
        if(json[i].info.hasOwnProperty('rank')) rank = json[i].info.rank.toString();
        else rank = '-1';

        batch.push({
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
    if(batch.length != 0) batches.push(batch);

    for(var i = 0; i < batches.length; i++) {
        console.log(`Inserting data batch ${i + 1}/${batches.length}`);
        await ddb.batchWriteItem({ RequestItems: { [TABLE]: batches[i] } }).promise();
    }
}

async function queryDynamoTable(movie, year, rating) {
    var params = {
        ExpressionAttributeValues: {
            ':y': {N: year},
            ':t': {S: movie},
            ':r': {N: rating}
        },
        FilterExpression: 'release_date = :y and begins_with (lowerCaseTitle, :t) and rating >= :r',
        TableName: TABLE
    }

    var raw = await ddb.scan(params).promise();
    var data = [];

    raw.Items.forEach(function(item, _, _) {
        data.push({
            title: item.title.S,
            year: item.release_date.N,
            rating: item.rating.N,
            rank: item.rank.N
        });
    });

    return data;
}

async function destroyDynamoTable() {
    var params = { TableName: TABLE };
    await ddb.deleteTable(params).promise();
}

function returnMessage(res, message, data) {
    body = {
        message: message,
        data: data
    };
    res.send(body);
    return;
}











app.listen(port, () => console.log(`Example app listening on port ${port}!`));



// Access key ID for AWS sdk:     AKIAZSSVVP446WY2R7GN
// Secret Access key for AWS sdk: TvPC1VtfEdNOWqvTK5qfcgFNfwqrOW4kei68W4Yj

// get it onto ec2
// lastly do a bit of rewriting

// https port at port 443