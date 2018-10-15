module.exports = function(s) {
   let working = []
   let max = 0
   for(i = 0; i < s.length; i++ ) {
     if (! working.includes(s[i])) {
       working.push(s[i])
     } else {
       if (working.length > max) {
         max = working.length
       }
       working.splice(0, working.indexOf(s[i]) + 1)
       working.push(s[i])
     }
   }
   return Math.max(max, working.length)
}





