module.exports = function(s) {
  //Whole string is palindrome
  if (isPalindrome(s)) {
    return s
  } 
  let longestPalindrome = s.substring(0,1)
  if (s.length >= 2 && isPalindrome(s.substring(0,2))) {
    longestPalindrome = s.substring(0,2)
  }
  let chars = s.split('')
  let dict = {}
  for(i=0; i < chars.length; i ++) {
    if (! dict[chars[i]]) {
      dict[chars[i]] = []
    }
    dict[chars[i]].push(i)
  }
  let propList = Object.getOwnPropertyNames(dict)
  if (propList.length < 3) {
    console.log('using dict')
    return useDict(dict, propList, longestPalindrome, s)
  } else {
    return walkTree(1, s, longestPalindrome)
  }
}  

let useDict = function(dict, propList, longestPalindrome, s) {
  propList.forEach((prop) =>{
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

let walkTree = function(startPoint, s, longestPalindrome) {
  if (startPoint >= s.length) return longestPalindrome
  if (longestPossible(startPoint, s) < longestPalindrome.length) {
    return walkTree(startPoint + 1, s, longestPalindrome)
  }
  let radius = 1
  let curPalindrome = longestPalindrome
  while(curPalindrome !== '') {
    if (longestPalindrome.length < curPalindrome.length) {
      longestPalindrome = curPalindrome
    }
    curPalindrome = palidromeIdentifier(s, startPoint, radius)
    radius++
  }
  return walkTree(startPoint + 1, s, longestPalindrome)
}

let palidromeIdentifier = function(s, startPoint, radius) {
  let testWordEven  
  let testWordOdd
  let oddStart = startPoint - radius
  let oddEnd = startPoint + radius + 1
  let evenStart = startPoint - radius + 1
  let evenEnd = startPoint + radius + 1

  if (oddStart >= 0 && oddEnd <= s.length) {
    testWordOdd = s.substring(oddStart, oddEnd)
  }
  
  if (evenStart >= 0 && evenEnd <= s.length) {
    testWordEven = s.substring(evenStart, evenEnd)
  }
  if (isPalindrome(testWordOdd)) {
    return testWordOdd
  }
  if (isPalindrome(testWordEven)) {
    return testWordEven
  }
  return ''
}  
  
let longestPossible = function(startPoint, s) {
  return ((s.length - (s.length - startPoint)) * 2) + 1
}
  
let isPalindrome = function(s) {
  return s && s === s.split('').reverse().join('')
}
