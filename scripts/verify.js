'use strict';

const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const Q=require('../qne/core/reconstruct.js');
const core=require('../build/qne_v13.node');
const dc=require('../build/qne_datacenter_v9.node');

assert.equal(typeof Q.mul,'function');
assert.equal(typeof core.mulBatch,'function');
assert.equal(typeof dc.dcBatch,'function');

const MASK=(1n<<64n)-1n;

function get(b,i){
  return b.readBigUInt64LE(i)|
    (b.readBigUInt64LE(i+8)<<64n);
}
function put(b,i,v){
  b.writeBigUInt64LE(v&MASK,i);
  b.writeBigUInt64LE(v>>64n,i+8);
}

const N=4096;
const a=crypto.randomBytes(N*16);
const b=crypto.randomBytes(N*16);
const out=typeof core.allocAligned==='function'
  ?core.allocAligned(N*16):Buffer.alloc(N*16);

put(a,0,0n);
put(a,16,1n);
put(a,32,(1n<<128n)-1n);

for(const threads of [1,4,16]){
  core.mulBatch(a,b,out,threads,'none','cached',0);
  for(let i=0;i<N;i++){
    assert.equal(
      get(out,i*16),
      Q.mul(get(a,i*16),get(b,i*16)),
      `QNE ${threads}T element ${i}`
    );
  }
  console.log(`QNE_${threads}T=PASS`);
}

const R=512,regions=10;
const seeds=crypto.randomBytes(R*regions*16);
const factors=crypto.randomBytes(regions*16);
const result=Buffer.alloc(R*16);

for(const threads of [1,8]){
  dc.dcBatch(
    seeds,factors,result,R,regions,
    threads,'none',0
  );

  for(let i=0;i<R;i++){
    let expected=0n;
    for(let j=0;j<regions;j++){
      expected^=Q.mul(
        get(seeds,16*(i*regions+j)),
        get(factors,16*j)
      );
    }
    assert.equal(
      get(result,16*i),expected,
      `DC ${threads}T requete ${i}`
    );
  }
  console.log(`DATACENTER_${threads}T=PASS`);
}

console.log('VERIFICATION_MATHEMATIQUE_COMPLETE=PASS');
