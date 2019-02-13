const LinkedList = function() {
  let length = 0
  let head = null

  let Node = function(element) {
    this.element = element
    this.next = null
  }

  let getLastNode = function(that) {
    let currentNode = that.head
    while(currentNode.next != null){
      currentNode = currentNode.next
    } 
    return currentNode
  }

  this.add = function(element) {
    let node = new Node(element)
    if (this.head == null) {
      this.head = node
    } else {
      let currentNode = getLastNode(this)
      currentNode.next = node
    }
    length ++
  }

  this.getFirst = function() {
    return this.head.element
  }

  this.getLast = function() {
    return getLastNode(this).element
  }

  this.getLength = function() {
    return length
  }

  this.remove = function(element) {
    let currentNode = this.head
    let previousNode = null
    while(currentNode.element != element) {
      if (currentNode.next == null) {
        throw new Error('Element not found in list.')
      }
      previousNode = currentNode
      currentNode = currentNode.next
    }
    if (previousNode == null) {
      this.head = currentNode.next
    } else {
      previousNode.next = currentNode.next
    }
    length --
  }

}

module.exports = LinkedList
