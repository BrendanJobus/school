let counter = 25

// every 1 second print to console
var interval = setInterval(function(){
    console.log(counter);
    counter--;
}, 1000)

// removes interval after 25 seconds
setTimeout(function() {
    clearInterval(interval);
}, 27000)

