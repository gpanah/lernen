const test = require('tape')

/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
const addTwoNumbers = function(l1, l2) {
  return recurse(l1, l2, 0)
}

const recurse = function(node1, node2, carry) {
  let sum = (node1 ? node1.val : 0) + (node2 ? node2.val : 0) + carry
  let newNode = new ListNode(sum > 9 ? sum - 10 : sum)
  let newCarry = sum > 9 ? 1 : 0
  let n1Next = node1 ? node1.next : null
  let n2Next = node2 ? node2.next : null
  if (n1Next || n2Next || newCarry > 0) {
    newNode.next = recurse(n1Next, n2Next, newCarry)
  }
  return newNode
}

const ListNode = function(val) {
  this.val = val
  this.next = null
}

test('add two numbers as linked lists', (t) =>{
  let l13 = new ListNode(3)
  let l12 = new ListNode(4)
  l12.next = l13
  let l11 = new ListNode(2)
  l11.next = l12

  let l23 = new ListNode(4)
  let l22 = new ListNode(6)
  l22.next = l23
  let l21 = new ListNode(5)
  l21.next = l22

  let lr3 = new ListNode(8)
  let lr2 = new ListNode(0)
  lr2.next = lr3
  let lr1 = new ListNode(7)
  lr1.next = lr2

  t.deepEqual(addTwoNumbers(l11, l21), lr1)
  t.end()
})

test('add two numbers as linked lists', (t) =>{
  let l11 = new ListNode(5)

  let l21 = new ListNode(5)

  let lr2 = new ListNode(1)
  let lr1 = new ListNode(0)
  lr1.next = lr2

  t.deepEqual(addTwoNumbers(l11, l21), lr1)
  t.end()
})
