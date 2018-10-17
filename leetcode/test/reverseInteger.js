let test = require('tape'),
  reverser = require('../algorithms/reverseInteger.js')



test('reverse 1', (t) => {
  t.equal(reverser(123), 321, 'simple num')
  t.equal(reverser(-123), -321, 'negative num')
  t.equal(reverser(1534236469), 0, 'overflow positive')
  t.end()
})
