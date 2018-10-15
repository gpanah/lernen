module.exports = function(s) {
  //Whole string is palindrome
  if (isPalindrome(s)) {
    return s
  } 
  let longestPalindrome = s.substring(0,1)
  if (s.length >= 2 && isPalindrome(s.substring(0,2))) {
    longestPalindrome = s.substring(0,2)
  }
  return walkTree(1, s, longestPalindrome)
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
  let distToEnd = s.length - startPoint
  let maxRadius = startPoint < distToEnd ? startPoint : distToEnd
  return (2 * maxRadius) + (s.length % 2 == 1 ? 1 : 2)
}
  
let isPalindrome = function(s) {
  return s && s === s.split('').reverse().join('')
}
