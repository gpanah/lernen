const test = require('tape'),
  LinkedList = require('../linkedlist.js')

test('Linked list instantiate', (t) => {
  let list = new LinkedList()
  t.equal(typeof list, 'object')
  t.end()
})

test('add', (t) => {
  let list = new LinkedList()
  list.add('john')
  t.end()
})

test('get first', (t) => {
  let element = 'john'
  let list = new LinkedList()
  list.add(element)
  t.equal(list.getFirst(), element, 'Able to retrieve first item')
  t.end()
})

test('get last', (t) => {
  let firstElement = 'john'
  let lastElement = 'jacob'
  let list = new LinkedList()
  list.add(firstElement)
  t.equal(list.getLast(), firstElement, 'Able to retrieve last item in list of 1.')
  list.add(lastElement)
  t.equal(list.getLast(), lastElement, 'Able to retrieve last item in list of 2.')
  t.end()
})

test('get last node is undefined', (t) => {
  let list = new LinkedList()
  try {
    list.getLastNode()
    t.fail('should have errored out for undefined')
  } catch(e) {
    t.equal(e.message, 'list.getLastNode is not a function')
  }
  t.end()
})

test('length', (t) => {
  let firstElement = 'john'
  let lastElement = 'jacob'
  let list = new LinkedList()
  list.add(firstElement)
  t.equal(list.getLength(), 1, 'List of length 1 correct.')
  list.add(lastElement)
  t.equal(list.getLength(), 2, 'List of length 2 correct.')
  t.end()
})

test('remove - first', (t) => {
  let firstElement = 'john'
  let lastElement = 'jacob'
  let list = new LinkedList()
  list.add(firstElement)
  list.add(lastElement)
  list.remove(firstElement)
  t.equal(list.getLength(), 1)
  t.equal(list.getFirst(), lastElement)
  t.equal(list.getLast(), lastElement)
  t.end()
})

test('remove - last', (t) => {
  let firstElement = 'john'
  let lastElement = 'jacob'
  let list = new LinkedList()
  list.add(firstElement)
  list.add(lastElement)
  list.remove(lastElement)
  t.equal(list.getLength(), 1)
  t.equal(list.getFirst(), firstElement)
  t.equal(list.getLast(), firstElement)
  t.end()
})

test('remove - middle', (t) => {
  let firstElement = 'john'
  let middleElement = 'jinglehiemer'
  let lastElement = 'jacob'
  let list = new LinkedList()
  list.add(firstElement)
  list.add(middleElement)
  list.add(lastElement)
  list.remove(middleElement)
  t.equal(list.getLength(), 2)
  t.equal(list.getFirst(), firstElement)
  t.equal(list.getLast(), lastElement)
  t.end()
})
