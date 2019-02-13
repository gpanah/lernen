const test = require('tape'),
  graph = require('../graph.js')


test('weighted graph', (t) => {

  let myGraph = new graph.WeightedDirectedGraph()
  myGraph.addEdge('start', 'B', 2)
  myGraph.addEdge('start', 'A', 5)
  myGraph.addEdge('A', 'C', 4)
  myGraph.addEdge('A', 'D', 2)
  myGraph.addEdge('B', 'A', 8)
  myGraph.addEdge('B', 'D', 7)
  myGraph.addEdge('C', 'D', 6)
  myGraph.addEdge('C', 'finish', 3)
  myGraph.addEdge('D', 'finish', 1)
  // myGraph.print()
  myGraph.dijkstra()
  t.end()
})
