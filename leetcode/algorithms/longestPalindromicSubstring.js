module.exports = function(s) {
  //Whole string is palindrome
  if (isPalindrome(s)) {
    return s
  } 
  let chars = s.split('')
  let dict = {}
  let longestPalindrome = s.substring(0,1)
  for(i=0; i < chars.length; i ++) {
    if (! dict[chars[i]]) {
      dict[chars[i]] = []
    }
    dict[chars[i]].push(i)
  }

  Object.getOwnPropertyNames(dict).forEach((prop) =>{
    for(i=0;i < dict[prop].length - 1;i++) {
      let start = dict[prop][i]
      let longestPossible = dict[prop][dict[prop].length -1] - start + 1
      if (longestPossible < longestPalindrome.length) {
        break
      }
      for(y=dict[prop].length -1; y > i; y--) {
        let end = dict[prop][y]
        let testWord = s.substring(start, end +1)
        if (isPalindrome(testWord)) {
          if (longestPalindrome.length < testWord.length) {
            longestPalindrome = testWord
          }
          break //this is the longest one possible 
        }
      }
    }
  })
  return longestPalindrome
}

let isPalindrome = function(s) {
  return s === s.split('').reverse().join('')
}
