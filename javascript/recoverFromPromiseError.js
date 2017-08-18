

let _x = 1


doIt = function() {
  return new Promise(every3rdTimeThrowError)
}


every3rdTimeThrowError = function(resolve, reject) {
    if (_x % 3 == 0) {
      _x += 1
      reject("3rd times the charm")
    } else {
      _x += 1
      resolve("Good")
    }
}

handleError = function(err){
  console.log('handling error: ' + err)
  recurse()
}

publish = function(goodMessage) {
  console.log('it was good: ' + goodMessage)
}

recurse = function() {
  doIt()
    .then(publish)
    .then(recurse)
    .catch(handleError)
}

recurse()
