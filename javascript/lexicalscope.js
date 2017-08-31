function foo(a) {

    var b = a * 2;

    function bar(c) {
        return [a, b, c ];
    }

    return bar( b * 3 )
}

res = foo( 2 )

console.assert(2 === res[0])
console.assert(4 === res[1])
console.assert(12 === res[2])

/* order doesn't matter...  `b` is still visible to bar2 */

function foo2(a) {

  function bar2(c) {
    return [a, b, c ];
  }

  var b = a * 2;

  return bar2( b * 3 )
}

res2 = foo2( 2 )

console.assert(2 === res2[0])
console.assert(4 === res2[1])
console.assert(12 === res2[2])

/*    */

function foo3() {
    function bar3(a) {
        i = 3; // changing the `i` in the enclosing scope's
               // for-loop
        console.log( a + i );
    }

    for (var i=0; i<10; i++) {
        console.log(`i is ${i}`)
        bar3( i * 2 ); // oops, inifinite loop ahead!
        console.log(`i is ${i}`)
    }
}

foo3();
