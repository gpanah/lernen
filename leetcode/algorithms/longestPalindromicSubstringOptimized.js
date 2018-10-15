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
  let radiiTree = getRadiiTree()
  let radius = 1
  while(radius > 0) {
    curPalindrome = palidromeIdentifier(s, startPoint, radius)
    radiiTree.update(radius, curPalindrome != '')
    if (longestPalindrome.length < curPalindrome.length) {
      longestPalindrome = curPalindrome
    }
    radius = radiiTree.getRadiusToAttempt(longestPossible(startPoint, s), longestPalindrome)
    
  }
  return walkTree(startPoint + 1, s, longestPalindrome)
}

let getRadiiTree = function() {
  return {
    valid: [],
    invalid:[],
    update: function(radius, succeeded) {
      let list = succeeded ? this.valid : this.invalid
      let i = 0
      for(i=0;i < list.length; i++) {
        if (list[i] > radius) {
          break
        }
      }
      list.splice(i, 0, radius)
    },
    getRadiusToAttempt: function(longestPossible, longestPalindrome) {
      if (longestPossible <= longestPalindrome.length) {
        return 0
      }
      let longestRadius = Math.trunc(longestPossible / 2) + 1
      if (this.invalid.length > 0 && longestPalindrome.length >= this.invalid[0] * 2) {
        return 0
      } 
      let upperBound
      let lowerBound
      if (this.invalid.length == 0) { 
        return longestRadius
      } else {
        upperBound = this.invalid[0]
        lowerBound = this.valid[this.valid.length -1]
      }
      if (upperBound - lowerBound == 1) {
        return 0
      }
      return Math.trunc((lowerBound + upperBound)/2)
    }
  }
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
