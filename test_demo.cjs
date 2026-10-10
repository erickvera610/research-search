const assert=require('node:assert/strict');
const {run}=require('./docs/demo.js');
assert.equal(run({query:'logistic classification'}).results[0].title,'Machine learning validation');assert.equal(run({query:'xyzabc'}).results.length,0);assert.throws(()=>run({query:' '}));
console.log('Browser demo behavior checks passed');

assert.equal(run({query:'machine learning validation'}).results[0].title,'Machine learning validation');
