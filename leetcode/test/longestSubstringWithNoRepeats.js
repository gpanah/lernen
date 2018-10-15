const lswnr = require('../algorithms/longestSubstringWithNoRepeats.js'),
  test = require('tape')

test('1', (t) => {

  t.equal(lswnr('bbbbb'),1)
  t.end()
})


test('2', (t) => {
  t.equal(lswnr('abcabcabc'),3)
  t.end()
})

test('3', (t) => {
  t.equal(lswnr('pwwkew'),3)
  t.end()
})

test('4', (t) => {
  t.equal(lswnr('dvdf'),3)
  t.end()
})

test('4', (t) => {
  t.equal(lswnr('abcadefa'),6)
  t.end()
})
