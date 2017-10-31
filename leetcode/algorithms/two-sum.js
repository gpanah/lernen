const test = require('tape')

const algorithm2 = function(nums, target) {
  let thing1
  for(thing1=0; thing1 < nums.length;thing1++) {
    let thing2 = nums.indexOf(target - nums[thing1])
    if (thing2 > -1 && thing2 != thing1) {
      return [thing1, thing2]
    }
  }
  return []
}

test('two sums', (t) =>{
  let target = 9
  let nums = [2, 7, 11, 15]
  t.deepEqual(algorithm2(nums, target), [0,1])
  t.end()
})
