// FASTEST
// module.exports = (nums1, nums2) => {
//   let res = []
//   let m = 0
//   let n = 0
//   let medianPos = []
//   let totalLength = nums1.length + nums2.length
//   if (totalLength % 2 == 0) {
//     let val = totalLength / 2
//     medianPos.push(val-1, val)
//   } else {
//     medianPos.push(Math.trunc(totalLength /2)) 
//   }
//   while(nums1.length > m && nums2.length > n) {
//     if (nums1[m] < nums2[n]) {
//       res.push(nums1[m])
//       m++
//     } else {
//       res.push(nums2[n])
//       n++
//     }
//   }
//   // add leftovers
//   while(nums1.length > m) {
//     res.push(nums1[m])
//     m++
//   }
//   while(nums2.length > n) {
//     res.push(nums2[n])
//     n++
//   }
//   let sum = 0
//   let i = 0
//   for(i=0; i < medianPos.length; i++) {
//     sum += res[medianPos[i]]
//   }
//   return sum / medianPos.length
// }



// ROUGHLY EQUAL
// module.exports = (nums1, nums2) => {
//   let res = []
//   let m = 0
//   let n = 0
//   let medianPos = []
//   let totalLength = nums1.length + nums2.length
//   if (totalLength % 2 == 0) {
//     let val = totalLength / 2
//     medianPos.push(val-1, val)
//   } else {
//     medianPos.push(Math.trunc(totalLength /2)) 
//   }
//   return perform(nums1, nums2, m, n, res, medianPos)
// }

// let perform = (nums1, nums2, m, n, res, medianPos) => {
//   if (n + m > medianPos[medianPos.length -1]) {
//     return calc(res, medianPos)
//   } else {
//     if (nums1[m] < nums2[n] || nums2[n] == undefined) {
//       res.push(nums1[m])
//       m++
//     } else {
//       res.push(nums2[n])
//       n++
//     }
//     return perform(nums1,nums2, m, n, res, medianPos)
//   }
// }

// let calc = (res, medianPos) => {
//   let sum = 0
//   let i = 0
//   for(i=0; i < medianPos.length; i++) {
//     sum += res[medianPos[i]]
//   }
//   return sum / medianPos.length
// }

// FASTEST
module.exports = (nums1, nums2) => {
  let even = false
  let lastPos = 0
  let totalLength = nums1.length + nums2.length
  if (totalLength % 2 == 0) {
    return perform(nums1, nums2, 0, 0, 0, 0, totalLength / 2, true)
  } else {
    return perform(nums1, nums2, 0, 0, 0, 0, Math.trunc(totalLength /2), false)
  }
}

let perform = (nums1, nums2, m, n, last, current, lastPos, even) => {
  if (n + m > lastPos) {
    if (even) {
      return (last + current) / 2
    } else {
      return current
    }
  } else {
    last = current
    if (nums1[m] < nums2[n] || nums2[n] == undefined) {
      current = nums1[m]
      m++
    } else {
      current = nums2[n]
      n++
    }
    return perform(nums1,nums2, m, n, last, current, lastPos,even)
  }
}
