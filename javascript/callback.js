
const validate = (number) => isNaN(number) ? 'Please pass a number' : null

const notEnoughMagic = function(number, callback) {
  let err = validate(number)
  let object = {
    number: number
  }
  callback(err)
  return object
}

const magic = function(number, callback) {
  let err
  if (isNaN(number)) {
    err = 'Please pass a number'
  }
  let object = {
    number: number,
  }
  Promise.resolve(1).then(() => callback(err))
  return object
}

var result = notEnoughMagic(4, function(err){
  if (err) {
    console.log(err)
  } else {
    console.log("in callback: " + result)
  }
})

console.log("at end: " + JSON.stringify(result))

var magicResult = magic(7, function(err){
  if (err) {
    console.log(err)
  } else {
    console.log("in magic callback: " + JSON.stringify(magicResult))
  }
})

console.log("at end: " + JSON.stringify(magicResult))
