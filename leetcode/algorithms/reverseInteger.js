// module.exports = (x) => {
//   let y = 1
//   if (x < 0) {
//     y = -1
//   }
//   let strArray = (x * y).toString().split('')
//   let reversedArray = strArray.reverse()
//   let newNum = parseInt(reversedArray.join('')) * y
//   if (newNum > (Math.pow(2, 31) - 1) || newNum < -(Math.pow(2,31))) {
//     return 0
//   }
//   return newNum
// }
// const INT_MAX = Math.pow(2, 31) -1,
//  INT_MIN = -1 * Math.pow(2, 31)

// module.exports = (x) => {
//   let rev = 0;
//   while (x != 0) {
//       let pop = x % 10;
//       x = Math.trunc(x/10);
//       if (rev > INT_MAX/10 || (rev == INT_MAX / 10 && pop > 7)) return 0
//       if (rev < INT_MIN/10 || (rev == INT_MIN / 10 && pop < -8)) return 0
//       rev = rev * 10 + pop;
//   }
//   return rev
// }
const INT_MAX = Math.pow(2, 31) -1,
 INT_MIN = -1 * Math.pow(2, 31)
module.exports = (x) => {
  let strArray = x.toString().split('')
  let i
  let res = ''
  let neg = ''
  for(i=strArray.length-1; i >= 0;i--) {
    if (strArray[i] !== '-') {
      res += strArray[i]
    } else {
      neg = '-'
    }
  }
  let newNum = parseInt(neg + res)
  if (newNum > INT_MAX || newNum < INT_MIN) {
    return 0
  }
  return newNum
}
