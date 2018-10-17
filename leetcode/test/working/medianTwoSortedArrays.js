const test = require('tape'), 
  medianFinder = require('../../algorithms/medianTwoSortedArrays.js')

test('find it', (t) => {
  t.equal(medianFinder([1,3], [2]), 2)
  t.equal(medianFinder([1,2], [3,4]), 2.5)
  t.equal(medianFinder([2], []), 2)
  t.equal(medianFinder([], [2]), 2)
  t.equal(medianFinder([5,15,25,35,45,55,65,75,85],[10,20,30,40,50,60,70,80,90]), 47.5)
  t.end()
}) 
