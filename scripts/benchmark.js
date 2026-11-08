'use strict';

const fs=require('node:fs');
const os=require('node:os');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const Q=require('../qne/core/reconstruct.js');
const core=require('../build/qne_v13.node');
const dc=require('../build/qne_datacenter_v9.node');

const get=(b,i)=>
  b.readBigUInt64LE(i)|
  (b.readBigUInt64LE(i+8)<<64n);

const median=a=>
  [...a].sort((x,y)=>x-y)[a.length>>1];

function bench(name,fn,ops){
  fn();
  const samples=[];

  for(let i=0;i<7;i++){
    const t=process.hrtime.bigint();
    fn();
    const seconds=Number(
      process.hrtime.bigint()-t
    )/1e9;
    samples.push(ops/seconds);
  }

  const value=median(samples);
  console.log(name+'='+Math.round(value)+'/s');

  return {
    median_per_sec:value,
    samples_per_sec:samples
  };
}

const K=4194304;
const a=crypto.randomBytes(K*16);
const b=crypto.randomBytes(K*16);
const out=typeof core.allocAligned==='function'
  ?core.allocAligned(K*16):Buffer.alloc(K*16);

const qne=bench(
  'QNE_V13_16T',
  ()=>core.mulBatch(
    a,b,out,16,'spread','stream',64
  ),
  K
);

for(let i=0;i<K;i+=32771){
  assert.equal(
    get(out,i*16),
    Q.mul(get(a,i*16),get(b,i*16)),
    `QNE bench ${i}`
  );
}
console.log('QNE_BENCH_ECHANTILLONS=PASS');

const R=32768,regions=10;
const seeds=crypto.randomBytes(R*regions*16);
const factors=crypto.randomBytes(regions*16);
const result=Buffer.alloc(R*16);

const dcSpeed=bench(
  'DATACENTER_V9_8T',
  ()=>dc.dcBatch(
    seeds,factors,result,R,regions,
    8,'compact',0
  ),
  R
);

for(let i=0;i<R;i+=127){
  let expected=0n;

  for(let j=0;j<regions;j++){
    expected^=Q.mul(
      get(seeds,16*(i*regions+j)),
      get(factors,16*j)
    );
  }

  assert.equal(
    get(result,i*16),expected,
    `Datacenter bench ${i}`
  );
}
console.log('DATACENTER_BENCH_ECHANTILLONS=PASS');

const previous={
  qne:1489531773,
  datacenter:88659448
};

const report={
  status:'PASS',
  date:new Date().toISOString(),
  cpu:os.cpus()[0]?.model,
  reference_commit:'41cce8919fca0565a374f28a1ae60ebfb5b3c5e4',
  benchmark:{
    qne_gf128_per_sec:qne,
    datacenter_requests_per_sec:dcSpeed
  },
  references_2026_11_07:previous,
  changes_percent:{
    qne:100*(qne.median_per_sec/previous.qne-1),
    datacenter:100*(
      dcSpeed.median_per_sec/previous.datacenter-1
    )
  },
  note:'Sept passages, comparaison historique indicative.'
};

fs.writeFileSync(
  process.env.QNE_REPORT,
  JSON.stringify(report,null,2)+'\n'
);

console.log('BENCHMARKS=PASS');
