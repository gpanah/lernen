/* global __dirname */
//Use the config from the production js file
const config = require('./webpack.config.js'),
  path = require('path')

//Set the public path.  This is the URL path in the browser.  Required for hot update.
config.output.publicPath = '/';

//bundle the js source files as extra scripts for in-browser debugging
config.devtool = 'eval';

//http://webpack.github.io/docs/webpack-dev-server.html#api
config.devServer = {
  //automatically causes the page to refresh.
  inline: true,
  //tells webpack where to serve files from.  Keep in mind that bundle.js is served from memory, so without this entered,
  //it won't be able to find the bundle.js
  contentBase: path.join(__dirname, 'build')
  //port is 8080 by default unless you otherwise specificy

};

//Basically re-exports the config for consumption the webpck dev server
module.exports = config;
