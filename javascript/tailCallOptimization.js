// function fib(n, tree, parent) {
//   if (tree == null) throw new Error('must have a tree object')
//   let me = {}
//   if (parent != null) {
//     parent[`fib-${n}`] = me
//   } else {
//     tree[`fib-${n}`] = me
//   }
//   if (n <= 1){
//     return n;
//   } else {
//     return fib(n-1, tree, me) + fib(n - 2, tree, me);
//   }
// }
//
// let callTree = {}
// console.log('*************** Recursive *******************')
// console.log(`Sum of first ${process.argv[2]} numbers in a fibonacci sequence is ${fib(process.argv[2] || 1, callTree)}`)
// console.log(JSON.stringify(callTree, null, '\t'))
//
//
// console.log('*************** Iterative *******************')
// function fibIter(n){
//   var a = 1, b = 0, temp;
//
//   while (n > 0){
//     temp = a;
//     a = a + b;
//     b = temp;
//     n--;
//   }
//
//   return b;
// }
// console.log(`Sum of first ${process.argv[2]} numbers in a fibonacci sequence is ${fibIter(process.argv[2] || 1)}`)
//
// console.log('*************** Iter/Recursive *******************')
//
function fibIterRecursive(n, a, b){
  console.log(fibIterRecursive.caller)
  if (n === 0) {
    return b;
  } else {
    return fibIterRecursive(n-1, a + b);
  }
};

function fib2(n, tree){
  console.log(fib2.caller)
  return fibIterRecursive(n, 1, 0);
}

let callTree2 = {}
console.log(`Sum of first ${process.argv[2]} numbers in a fibonacci sequence is ${fib2(process.argv[2] || 1, callTree2)}`)
console.log(JSON.stringify(callTree2, null, '\t'))

function fib3(n) {
  console.log(fib3.caller)
  if (n <= 1){
    return n;
  } else {
    return fib3(n-1) + fib3(n - 2);
  }
}

fib3(process.argv[2] || 1)
