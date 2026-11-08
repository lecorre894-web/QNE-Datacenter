#!/usr/bin/env node
'use strict';
/*
  QNE V2 -- extension ADDITIVE du depot QNE
  Validation des sauts GF(2^128) + RK4 Lorenz-96 sans allocations par pas.
  Node.js >= 18, aucune dependance externe.
  Ne modifie et ne charge aucun fichier canonique.
*/
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const W = 128, LOW = 0x87, MASK128=(1n<<128n)-1n;
const P128=(1n<<128n)|BigInt(LOW);
// Polynome reciproque : caracteristique du LFSR a decalage vers la droite.
const LOW_RECIP=(1n<<127n)|(1n<<126n)|(1n<<121n)|1n;
const P_RECIP=(1n<<128n)|LOW_RECIP;
const report = { schema: 'QNE-V2/1', created: new Date().toISOString(),
  node: process.version, arch: process.arch, platform: process.platform,
  cpu: os.cpus()[0]?.model || 'inconnu', notes: [
    'Les benchmarks sont locaux a cette machine.',
    'EXACT signifie identite des operations dans GF(2^128), pas precision infinie du RK4.',
    'La reduction de E modulo 128 necessite le test d irreductibilite de P128.'
  ], tests: [], benchmarks: {} };
function check(ok, label, details='') {
  const item={ name: label, passed: Boolean(ok), details };
  report.tests.push(item);
  console.log(`${ok?'OK':'ECHEC'} | ${label}${details?' | '+details:''}`);
  if (!ok) throw Error('Validation echouee: '+label);
}
function fromBig(v) {
  return new Uint32Array(Array.from({length:4}, (_,i)=> Number((v>>BigInt(i*32))&0xffffffffn)));
}
function toBig(p) {
  return BigInt(p[0]) | (BigInt(p[1])<<32n) | (BigInt(p[2])<<64n) | (BigInt(p[3])<<96n);
}
// Reference INDEPENDANTE, a base de BigInt : multiplication dans GF(2)[x]/P128.
function mulRef(a,b,lowPoly=BigInt(LOW)) {
  let result=0n;
  for(let i=0;i<W;i++) {
    if(b&1n) result ^= a;
    b >>= 1n;
    const overflow=(a>>127n)&1n;
    a=((a<<1n)&MASK128)^(overflow?lowPoly:0n);
  }
  return result;
}
function polyDegree(v){return v? v.toString(2).length-1 : -1;}
function polyRem(a,b){let d=polyDegree(b);if(d<0)throw Error('division par zero');while(polyDegree(a)>=d)a^=b<<BigInt(polyDegree(a)-d);return a;}
function polyGcd(a,b){while(b){[a,b]=[b,polyRem(a,b)];}return a;}
const spread16=new Uint16Array(256);
for(let b=0;b<256;b++){let s=0;for(let k=0;k<8;k++)if((b>>>k)&1)s|=(1<<(2*k));spread16[b]=s;}
function shl5(a,n){const out=new Uint32Array(5);out[0]=(a[0]<<n)>>>0;for(let i=1;i<4;i++)out[i]=((a[i]<<n)|(a[i-1]>>>(32-n)))>>>0;out[4]=a[3]>>>(32-n);return out;}
// Carre exact repris de la methode LUT polyjump.js : aucune generation de grands entiers.
function squareFast(poly) {
  const wide=new Uint32Array(8);
  for(let bi=0;bi<16;bi++){
    const byte=(poly[bi>>2]>>>((bi&3)*8))&255;
    wide[bi>>1]=(wide[bi>>1]|(spread16[byte]<<((bi&1)*16)))>>>0;
  }
  const hi=[wide[4],wide[5],wide[6],wide[7]];
  const f1=shl5(hi,1),f2=shl5(hi,2),f7=shl5(hi,7),fold=new Uint32Array(5);
  for(let i=0;i<4;i++) fold[i]=hi[i];
  for(let i=0;i<5;i++)fold[i]=(fold[i]^f1[i]^f2[i]^f7[i])>>>0;
  const hi2=fold[4]&255,c=(hi2^(hi2<<1)^(hi2<<2)^(hi2<<7))>>>0;
  return new Uint32Array([(wide[0]^fold[0]^c)>>>0,(wide[1]^fold[1])>>>0,
    (wide[2]^fold[2])>>>0,(wide[3]^fold[3])>>>0]);
}
function mulX32(p) {
  const carry=p[3]>>>31;
  p[3]=((p[3]<<1)|(p[2]>>>31))>>>0;
  p[2]=((p[2]<<1)|(p[1]>>>31))>>>0;
  p[1]=((p[1]<<1)|(p[0]>>>31))>>>0;
  p[0]=(p[0]<<1)>>>0;
  if(carry)p[0]^=LOW;
  return p;
}
// LFSR de polyjump.js : base transformant un polynome en etat vectoriel.
const vmask=new Uint32Array(4);
for(const tap of [128,7,2,1]){const bit=tap-1;vmask[bit>>>5]|=(1<<(bit&31))>>>0;}
function advanceLfsr(v){const lsb=v[0]&1;for(let i=0;i<4;i++)v[i]=((v[i]>>>1)|(i<3?((v[i+1]&1)<<31):0))>>>0;if(lsb)for(let i=0;i<4;i++)v[i]^=vmask[i];return v;}
const basis=[];
{let s=new Uint32Array([1,0,0,0]);for(let i=0;i<W;i++){basis.push(s.slice());advanceLfsr(s);}}
function stateFromPoly(poly) {
  const out=new Uint32Array(4);
  for(let i=0;i<W;i++)if((poly[i>>>5]>>>(i&31))&1){const b=basis[i];for(let j=0;j<4;j++)out[j]^=b[j];}
  return out;
}
const frobeniusCache=[];
{let x=new Uint32Array([2,0,0,0]);for(let e=0;e<W;e++){frobeniusCache.push(x);x=squareFast(x);} }
// Cache CORRIGE du LFSR original : quotient par le polynome reciproque,
// et non le polynome LOW=0x87 utilise pour la representation GCM.
const lfsrCache=[];
const recipPolyCache=[];
{let x=2n;for(let e=0;e<W;e++){const poly=fromBig(x);recipPolyCache.push(poly);lfsrCache.push(stateFromPoly(poly));x=mulRef(x,x,LOW_RECIP);} }
// Entree E : bigint, entier decimal (chaine) ou entier JS sur.
function expRemainder128(E){
  if(typeof E==='number') {
    if(!Number.isSafeInteger(E)||E<0) throw RangeError('E non entier sur');
    return E%128;
  }
  if(typeof E==='bigint') {if(E<0n)throw RangeError('E negatif');return Number(E%128n);}
  if(typeof E==='string' && /^\d+$/.test(E)) {
    let rem=0;for(let i=0;i<E.length;i++)rem=(rem*10+E.charCodeAt(i)-48)%128;
    return rem;
  }
  throw TypeError('E doit etre un entier positif ou une chaine de chiffres');
}
function at2powField(E){return frobeniusCache[expRemainder128(E)].slice();}
// Etat du LFSR de polyjump.js apres 2^E pas depuis le vecteur e0.
function at2pow(E){return lfsrCache[expRemainder128(E)].slice();}
// Generalisation : saut exact du LFSR a partir de tout etat 128 bits.
// Cout O(128) pas de registre, independant de la grandeur de E.
function jumpSeed2pow(E,seed){
  if(!(seed instanceof Uint32Array)||seed.length!==4)throw TypeError('seed: Uint32Array(4)');
  const coeff=recipPolyCache[expRemainder128(E)];
  const cur=seed.slice(),out=new Uint32Array(4);
  for(let i=0;i<128;i++){
    if((coeff[i>>>5]>>>(i&31))&1)for(let j=0;j<4;j++)out[j]^=cur[j];
    advanceLfsr(cur);
  }
  return out;
}
function hashState(state) {
  // Hachage observatoire, pas cryptographique.
  let h=2166136261>>>0;
  for(let i=0;i<state.length;i++){const v=Math.round(state[i]*1e6);h=Math.imul(h^(v>>>0),16777619)>>>0;}
  return h.toString(16).padStart(8,'0');
}
function linearTests(){
  console.log('\n=== JUMEAU LINEAIRE : validation GF(2^128) ET LFSR ===');
  // Test de Rabin d irreductibilite des deux polynomes.
  for(const [name,poly,low] of [['p=0x87',P128,BigInt(LOW)],['p reciproque LFSR',P_RECIP,LOW_RECIP]]) {
    let x=2n, at64;
    for(let e=1;e<=128;e++){x=mulRef(x,x,low);if(e===64)at64=x;}
    check(x===2n && polyGcd(at64^2n,poly)===1n, 'Irreductibilite '+name+' (Rabin)');
  }
  let rnd=0x8adf3402;
  const next=()=>{rnd^=rnd<<13;rnd^=rnd>>>17;rnd^=rnd<<5;return rnd>>>0;};
  for(let t=0;t<32;t++){
    const p=new Uint32Array([next(),next(),next(),next()]);
    check(toBig(squareFast(p))===mulRef(toBig(p),toBig(p)), `Carre LUT vs reference BigInt #${t+1}`);
  }
  for(let E=0;E<=12;E++){
    let field=new Uint32Array([1,0,0,0]);
    for(let i=0;i<2**E;i++)mulX32(field);
    check(toBig(at2powField(E))===toBig(field),`Saut champ GF(2^128) 2^${E} vs ${2**E} pas directs`);
    let vec=new Uint32Array([1,0,0,0]);for(let i=0;i<2**E;i++)advanceLfsr(vec);
    check(toBig(at2pow(E))===toBig(vec),`Saut LFSR CORRIGE 2^${E} vs LFSR original`);
  }
  // Generalisation sur graines arbitraires, et pas seulement la base e0.
  for(const seed of [
    new Uint32Array([0xffffffff,0x01020304,0xdeadbeef,0xf1e2d3c4]),
    new Uint32Array([0,1,0,0]),
    new Uint32Array([123,456,789,987654321])
  ]) for(const E of [0,3,7,10]){
    const direct=seed.slice();for(let i=0;i<2**E;i++)advanceLfsr(direct);
    check(toBig(jumpSeed2pow(E,seed))===toBig(direct),
      `Saut de graine arbitraire ${Array.from(seed).join(':')} / 2^${E}`);
  }
  // Regression de l anomalie presente dans polyjump.js : l ancien
  // quotient LOW=0x87 n est pas le polynome caracteristique de son LFSR.
  let oldReference=new Uint32Array([1,0,0,0]);
  for(let i=0;i<128;i++)advanceLfsr(oldReference);
  const oldWrong=stateFromPoly(at2powField(7));
  check(toBig(oldWrong)!==toBig(oldReference),
    'Defaut ancien polyjump.js reproduit a 2^7 (comparaison divergente)');
  // Test long distinct des petits sauts, egal a celui utilise par polyjump.js.
  let longDirect=new Uint32Array([1,0,0,0]);
  for(let i=0;i<(1<<20);i++)advanceLfsr(longDirect);
  check(toBig(at2pow(20))===toBig(longDirect),'Saut LFSR 2^20 vs 1 048 576 pas directs');
  check(toBig(stateFromPoly(at2powField(20)))!==toBig(longDirect),
    'Defaut ancien polyjump.js reproduit aussi a 2^20');
  check(toBig(at2pow(0))===toBig(at2pow(128)),'Frobenius LFSR E=0 et E=128');
  check(toBig(at2pow('1'+'0'.repeat(100)))===toBig(at2pow(BigInt('1'+'0'.repeat(100)))),
    'Exposant decimal 10^100 sans BigInt geant');
  const big=at2pow('1'+'0'.repeat(100));
  check(toBig(big)!==1n,'Le saut geant ne calcule pas carre(1)');
  const batch=20000;
  for(let i=0;i<1000;i++)at2pow(10n**100n);
  const start=performance.now();let chk=2166136261>>>0;
  for(let i=0;i<batch;i++){const p=at2pow(BigInt(i)*1000000000000000n+99999999999999n);chk=Math.imul(chk^p[i%4],16777619)>>>0;}
  const ms=performance.now()-start;
  report.benchmarks.linear={operation:'Etat du LFSR original apres 2^E pas', calls:batch,
    total_ms:ms, mean_us:ms*1000/batch, checksum:chk>>>0,
    memory_cache_bytes:W*4*4, polynome:'x^128+x^127+x^126+x^121+1'};
  console.log(`MESURE locale | ${batch} sauts LFSR, ${(ms*1000/batch).toFixed(3)} us/saut, empreinte ${chk>>>0}`);
  console.log(`Etat a 2^(10^100) : ${Array.from(big).map(v=>v.toString(16).padStart(8,'0')).reverse().join('')}`);
}

const FORCE=8,DT=0.01;
function initial(N){const x=new Float64Array(N);for(let i=0;i<N;i++)x[i]=FORCE+(i%7-3)*0.001;return x;}
// Baseline strictement conforme au coeur de calcul_max.js (a buffers fixes).
function baseline(N) {
  if(N<4)throw RangeError('Lorenz-96 exige N>=4');
  const x=initial(N),acc=new Float64Array(N),tmp=new Float64Array(N),k=new Float64Array(N);
  const deriv=(a,out)=>{for(let i=0;i<N;i++){
    const ip1=i+1<N?i+1:0,im1=i>0?i-1:N-1,im2=i>1?i-2:i+N-2;
    out[i]=(a[ip1]-a[im2%N])*a[im1]-a[i]+FORCE;
  }};
  return {x,step(){
    deriv(x,k);for(let i=0;i<N;i++){acc[i]=k[i];tmp[i]=x[i]+DT/2*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++){acc[i]+=2*k[i];tmp[i]=x[i]+DT/2*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++){acc[i]+=2*k[i];tmp[i]=x[i]+DT*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++)x[i]+=DT/6*(acc[i]+k[i]);
  }};
}
// V2 : memes 4 buffers, specialisation de la topologie en anneau :
// aucun modulo, aucune branche conditionnelle sur i dans la boucle principale.
function optimized(N) {
  if(N<4)throw RangeError('Lorenz-96 exige N>=4');
  const x=initial(N),acc=new Float64Array(N),tmp=new Float64Array(N),k=new Float64Array(N);
  function deriv(a,out){
    out[0]=(a[1]-a[N-2])*a[N-1]-a[0]+FORCE;
    out[1]=(a[2]-a[N-1])*a[0]-a[1]+FORCE;
    for(let i=2;i<N-1;i++)out[i]=(a[i+1]-a[i-2])*a[i-1]-a[i]+FORCE;
    out[N-1]=(a[0]-a[N-3])*a[N-2]-a[N-1]+FORCE;
  }
  return {x,step(){
    deriv(x,k);for(let i=0;i<N;i++){acc[i]=k[i];tmp[i]=x[i]+DT/2*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++){acc[i]+=2*k[i];tmp[i]=x[i]+DT/2*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++){acc[i]+=2*k[i];tmp[i]=x[i]+DT*k[i];}
    deriv(tmp,k);for(let i=0;i<N;i++)x[i]+=DT/6*(acc[i]+k[i]);
  }};
}
function maxError(a,b){let worst=0;for(let i=0;i<a.length;i++){worst=Math.max(worst,Math.abs(a[i]-b[i]));}return worst;}
function benchOne(f,N,steps) {
  const obj=f(N);
  // Echauffement inclus hors de la mesure, memes conditions pour les variantes.
  for(let i=0;i<5;i++)obj.step();
  const start=performance.now();for(let i=0;i<steps;i++)obj.step();
  const ms=performance.now()-start;
  return { ms, variables_per_s: N*steps*1000/ms,
    steps_per_s:steps*1000/ms, checksum:hashState(obj.x) };
}
function median(values){const a=values.slice().sort((a,b)=>a-b);return a[Math.floor(a.length/2)];}
function nonlinearTests(large=false){
  console.log('\n=== JUMEAU NON LINEAIRE : validation Lorenz-96 RK4 ===');
  for(const N of [4,5,7,40,100,4096]) {
    const a=baseline(N),b=optimized(N);
    for(let k=0;k<20;k++){a.step();b.step();}
    const err=maxError(a.x,b.x);
    check(err<=1e-12, `RK4 V2 vs calcul_max.js pour N=${N}`,`erreur max ${err.toExponential(2)}`);
  }
  const sizes = large?[40,4096,65536,262144,1048576]:[40,4096,65536,262144];
  const result=[];
  for(const N of sizes){
    const steps=N<=4096?400:(N<=65536?80:(N<=262144?30:10));
    // Alternance des conditions par repetition, medians moins sensibles aux fluctuations.
    const a=[],b=[];for(let repeat=0;repeat<5;repeat++){
      if(repeat%2===0){a.push(benchOne(baseline,N,steps));b.push(benchOne(optimized,N,steps));}
      else{b.push(benchOne(optimized,N,steps));a.push(benchOne(baseline,N,steps));}
    }
    const oldMs=median(a.map(v=>v.ms)),newMs=median(b.map(v=>v.ms));
    const speedup=oldMs/newMs;
    const row={N,steps,repetitions:5,baseline_median_ms:oldMs,v2_median_ms:newMs,
      baseline_mvariables_s:N*steps/oldMs/1000,v2_mvariables_s:N*steps/newMs/1000,
      speedup,allocated_bytes:4*N*8};
    result.push(row);
    console.log(`N=${N} | original ${row.baseline_mvariables_s.toFixed(2)} M maj/s | V2 ${row.v2_mvariables_s.toFixed(2)} M maj/s | x${speedup.toFixed(2)}`);
  }
  report.benchmarks.nonlinear=result;
}
function main(){
  const args=new Set(process.argv.slice(2));
  if(args.has('--help')||args.has('-h')){
    console.log('Usage: node qne_v2.js [--linear | --nonlinear] [--large] [--no-save]');return;
  }
  console.log('QNE V2 | extension additive | aucune modification du depot canonique');
  const runLinear=!args.has('--nonlinear'),runNonlinear=!args.has('--linear');
  try {
    if(runLinear)linearTests();
    if(runNonlinear)nonlinearTests(args.has('--large'));
    report.status='PASSED';
  } catch(e) {report.status='FAILED';report.error=String(e.stack||e);console.error('ECHEC :',e.message);process.exitCode=1;}
  finally {
    if(!args.has('--no-save')) {
      const file=path.join(process.cwd(),'QNE_V2_RESULTATS.json');
      fs.writeFileSync(file,JSON.stringify(report,null,2)+'\n');
      console.log('Rapport : '+file);
    }
  }
  console.log('STATUT : '+report.status);
}
if(require.main===module)main();
module.exports={squareFast,at2pow,at2powField,jumpSeed2pow,mulRef,baseline,optimized,linearTests,nonlinearTests};
