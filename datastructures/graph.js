const WeightedDirectedGraph = function() {
  this.graph = {}
}

WeightedDirectedGraph.prototype.addEdge = function(start, end, weight) {
  if (!this.graph[start]) {
    this.graph[start] = {}
  }
  this.graph[start][end] = weight
}

WeightedDirectedGraph.prototype.print = function() {
  Object.getOwnPropertyNames(this.graph).forEach((start) => {
    Object.getOwnPropertyNames(this.graph[start]).forEach((end) => {
      console.log(`${start} ---${this.graph[start][end]}---> ${end}`)
    })
  });
}

const lowestCostNode = (costs, processed) => {
  return Object.keys(costs).reduce((lowest, node) => {
    if (lowest === null || costs[node] < costs[lowest]) {
      if (!processed.includes(node)) {
        lowest = node
      }
    }
    return lowest
  }, null)
};

WeightedDirectedGraph.prototype.dijkstra = function() {
  const costs = Object.assign({finish: Infinity}, this.graph.start)
  const parents = {finish: null}

  for (let child in this.graph.start) {  // add children of start node
    parents[child] = 'start'
  }

  const processed = []
  let node = lowestCostNode(costs, processed);
  while (node) {

    let cost = costs[node];

    let children = this.graph[node];

    for (let n in children) {
      let newCost = cost + children[n];
      if (!costs[n]) {
        costs[n] = newCost;
        parents[n] = node;
      }
      if (costs[n] > newCost) {
        costs[n] = newCost;
        parents[n] = node;
      }
    }

    processed.push(node);
    console.log(costs)
    node = lowestCostNode(costs, processed);

  }

  let optimalPath = ['finish'];

  let parent = parents.finish;

  while (parent) {
    optimalPath.push(parent);
    parent = parents[parent];
  }
  
  optimalPath.reverse();  // reverse array to get correct order

  const results = {
    distance: costs.finish,
    path: optimalPath
  };

  console.log(results)
}





module.exports.WeightedDirectedGraph = WeightedDirectedGraph
