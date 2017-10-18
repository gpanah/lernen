/* global __dirname */
//__dirname - global object - the name of the directory tha the currently executing script resides in
//path - for working with directories.  Understands differences between OS's.
//Every webpack.config.js needs an entry and an output.
const path = require('path')

module.exports = {
  //says index.js is my entry point for determining all my dependencies
  entry: path.join(__dirname, 'src/index.js'),
  //indicates the directory and file name under which to save the bundled js
  output: {
    path: path.join(__dirname, 'build'),
    filename: 'bundle.js'
  },
  module: {
    //for each .js, execute the babel-loader, which will transpile es6 as needed
    loaders:[{
      test: /\.js?$/,
      loader: 'babel-loader',
      exclude: /(node_modules)/
      //you can do it this way, or you can configure a .babelrc, which is what I have done
      // query: {
      //   presets: ['es2015']
      // }
    }, {
      test: /\.less$/,
      loader: 'style-loader!css-loader!less-loader'
    }]
  }
};
