function countDown(i) {
    return promise = new Promise( (resolve, reject) => {
        console.log(i);
        i -= 5

        if (i >= 0) {
            setTimeout( () => {
                resolve(countDown(i));
            }, 5000);
        } else {
            resolve('counter finished');
        }
    })
}

let counter = countDown(25);
counter.then( (msg) => {
    console.log(msg);
});

// // every 1 second print to console
// let interval = setInterval(function(){
//     console.log(counter);
//     counter--;
// }, 1000)

// // removes interval after 25 seconds
// setTimeout(function() {
//     clearInterval(interval);
// }, 27000)

