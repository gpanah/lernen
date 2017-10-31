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

};

const ListNode = function(val) {
  this.val = val
  this.next = null
}

const LinkedList = function() {
  this._length = 0
  this.head = null
}

LinkedList.prototype.add = function(value) {
  let node = new ListNode(value)
  let currentNode = this.head

  if (!currentNode) {
    this.head = node
    this._length++
    return node
  }

  while(currentNode.next) {
    currentNode = currentNode.next
  }

  currentNode.next = node
  this._length ++

  return node
}

const buildList = function(number) {

}



test('add two numbers as linked lists', (t) =>{
  let l13 = new ListNode(3)
  let l12 = new ListNode(4)
  l12.next = l13
  let l11 = new ListNode(2)
  l11.next = l12

  let l23 = new ListNode(4)
  let l22 = new ListNode(6)
  l12.next = l23
  let l21 = new ListNode(5)
  l11.next = l22

  t.deepEqual(addTwoNumbers(l11, l21), )
  t.end()
})
