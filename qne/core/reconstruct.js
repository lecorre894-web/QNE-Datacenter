'use strict';
// QNE V15: exact, lossless reconstruction with explicit fallback for arbitrary bytes.
// GF128 engine is an unmodified copy of QNE V2. No arbitrary code deserialization.
const crypto=require('node:crypto');
const QNE=require('../engine/qne_v2.js');
const MASK=(1n<<128n)-1n, MAX_READ=1024*1024;
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
const hex=v=>'0x'+v.toString(16).padStart(32,'0');
function nat(x){let n;if(typeof x==='bigint')n=x;else if(typeof x==='number' && Number.isSafeInteger(x))n=BigInt(x);else if(typeof x==='string' && /^\d+$/.test(x))n=BigInt(x);else throw Error('Entier naturel requis');if(n<0n)throw Error('Entier negatif');return n;}
function fieldValue(x){if(typeof x==='string'&&/^0x[0-9a-f]{1,32}$/i.test(x))return BigInt(x);return nat(x);}
function nonzero(n,label='Valeur'){n=nat(n);if(n<=0n)throw Error(label+' doit etre positif');return n;}
function toWords(v){return Uint32Array.from({length:4},(_,i)=>Number((v>>(BigInt(i)*32n))&0xffffffffn));}
function fromWords(a){return BigInt(a[0])|(BigInt(a[1])<<32n)|(BigInt(a[2])<<64n)|(BigInt(a[3])<<96n);}
const mul=(a,b)=>QNE.mulRef(a,b);
const sq=a=>fromWords(QNE.squareFast(toWords(a)));
const mulx=v=>((v<<1n)&MASK)^((v>>127n)?0x87n:0n);
function pow(a,e){e=nat(e);let v=1n;while(e){if(e&1n)v=mul(v,a);e>>=1n;if(e)a=sq(a);}return v;}
function combine(a,b){return {p:mul(a.p,b.p),s:a.s^mul(a.p,b.s)};}
function geometrical(multiplier,n){n=nat(n);let a={p:1n,s:0n},b={p:multiplier,s:1n};for(;n;n>>=1n){if(n&1n)a=combine(a,b);if(n>1n)b={p:sq(b.p),s:b.s^mul(b.p,b.s)};}return a;}
function gf128(seed,nbytes){seed=fieldValue(seed);nbytes=nat(nbytes);if(seed>MASK||nbytes%16n!==0n)throw Error('GF128 requiert graine <= 128 bits et taille multiple de 16');return {kind:'gf128',nbytes:String(nbytes),seed:hex(seed),polynomial:'0x87',rule:'cell[i]=seed*x^i in GF(2^128)'};}
function lorenz96(N,steps,expectedHash){
 if(!Number.isInteger(N)||N<4||N>4096||!Number.isInteger(steps)||steps<0||steps>256)throw Error('Lorenz96 borne N=4..4096, pas 0..256');
 if(typeof expectedHash!=='string'||!(/^[a-f0-9]{64}$/).test(expectedHash))throw Error('Empreinte SHA-256 finale requise');
 return {kind:'lorenz96',nbytes:String(N*8),N,steps,parameters:{F:8,dt:0.01,initial:'QNE-V2-initial'},sha256:expectedHash};
}
function generateLorenz(d){if(!Number.isInteger(d.N)||d.N<4||d.N>4096||!Number.isInteger(d.steps)||d.steps<0||d.steps>256||BigInt(d.N)*8n!==size(d))throw Error('Lorenz96 descripteur invalide');
 const engine=QNE.optimized(d.N);for(let i=0;i<d.steps;i++)engine.step();
 const out=Buffer.alloc(d.N*8);for(let i=0;i<d.N;i++)out.writeDoubleLE(engine.x[i],i*8);
 if(sha(out)!==d.sha256)throw Error('Etat Lorenz96 divergent: pas de reconstruction fiable');return out;
}
function literal(buf){if(!Buffer.isBuffer(buf))throw Error('Buffer requis');return {kind:'literal',nbytes:String(buf.length),data:buf.toString('base64'),sha256:sha(buf)};}
function periodic(pattern,nbytes){if(!Buffer.isBuffer(pattern)||pattern.length<1||pattern.length>65536)throw Error('Motif 1..65536 octets requis');return {kind:'periodic',nbytes:String(nonzero(nbytes)),pattern:pattern.toString('base64')};}
function fill(byte,nbytes){if(!Number.isInteger(byte)||byte<0||byte>255)throw Error('Octet 0..255 attendu');return {kind:'fill',nbytes:String(nonzero(nbytes)),byte};}
function patches(base,changes){if(!Array.isArray(changes)||changes.length>4096)throw Error('Nombre de correctifs invalide');const lim=size(base);let last=0n;let res=[];for(const ch of changes){const at=nat(ch.at),b=Buffer.isBuffer(ch.data)?ch.data:Buffer.from(ch.data||'', 'base64');if(!b.length||at<last||at+BigInt(b.length)>lim)throw Error('Correctifs hors limites ou chevauchants');res.push({at:String(at),data:b.toString('base64')});last=at+BigInt(b.length);}return {kind:'patches',nbytes:String(lim),base,changes:res};}
function size(d){if(!d||typeof d!=='object'||!['literal','periodic','fill','gf128','patches','lorenz96'].includes(d.kind))throw Error('Descripteur inconnu');return nat(d.nbytes);}
function validDescriptor(d){const n=size(d);if(d.kind==='literal'){const b=Buffer.from(d.data||'','base64');if(BigInt(b.length)!==n||sha(b)!==d.sha256)throw Error('Litteral altere');}
 if(d.kind==='periodic'&&!Buffer.from(d.pattern||'','base64').length)throw Error('Motif vide');
 if(d.kind==='fill'&&(!Number.isInteger(d.byte)||d.byte<0||d.byte>255))throw Error('Octet invalide');
 if(d.kind==='gf128'&&(n%16n!==0n||fieldValue(d.seed)>MASK||d.polynomial!=='0x87'))throw Error('Graine GF invalide');
 if(d.kind==='lorenz96'&&(BigInt(d.N)*8n!==n||d.N<4||d.N>4096||d.steps<0||d.steps>256||!(/^[a-f0-9]{64}$/).test(d.sha256)))throw Error('Etat Lorenz96 invalide');
 if(d.kind==='patches'){if(size(d.base)!==n)throw Error('Taille de base incoherente');let last=0n;for(const c of d.changes){const at=nat(c.at),b=Buffer.from(c.data||'','base64');if(!b.length||at<last||at+BigInt(b.length)>n)throw Error('Correctif incorrect');last=at+BigInt(b.length);}}
 return true;
}
function read(d,start,length){const n=size(d),at=nat(start);if(!Number.isSafeInteger(length)||length<0||length>MAX_READ||at+BigInt(length)>n)throw Error('Lecture hors limites 0..1MiB');const out=Buffer.alloc(length);if(length===0)return out;
 if(d.kind==='lorenz96'){const b=generateLorenz(d);b.copy(out,0,Number(at),Number(at)+length);return out;}
 if(d.kind==='literal'){Buffer.from(d.data,'base64').copy(out,0,Number(at),Number(at)+length);return out;}
 if(d.kind==='fill'){out.fill(d.byte);return out;}
 if(d.kind==='periodic'){const pattern=Buffer.from(d.pattern,'base64');for(let i=0;i<length;i++)out[i]=pattern[Number((at+BigInt(i))%BigInt(pattern.length))];return out;}
 if(d.kind==='gf128'){let ci=at/16n,inCell=Number(at%16n),v=mul(fieldValue(d.seed),pow(2n,ci)),written=0;
  while(written<length){const bytes=Buffer.allocUnsafe(16);for(let i=0;i<16;i++)bytes[i]=Number((v>>(8n*BigInt(i)))&255n);const take=Math.min(16-inCell,length-written);bytes.copy(out,written,inCell,inCell+take);written+=take;inCell=0;v=mulx(v);}return out;}
 if(d.kind==='patches'){const base=read(d.base,at,length);for(const c of d.changes){const off=nat(c.at),b=Buffer.from(c.data,'base64');const left=off>at?off:at,right=off+BigInt(b.length)<at+BigInt(length)?off+BigInt(b.length):at+BigInt(length);if(left<right)b.copy(base,Number(left-at),Number(left-off),Number(right-off));}return base;}
 throw Error('Type non gere');
}
function discover(buf){if(!Buffer.isBuffer(buf)||buf.length>1024*1024)throw Error('Buffer <1 MiB requis pour encodeAuto');let variants=[literal(buf)];if(!buf.length)return variants[0];
 if(buf.every(v=>v===buf[0]))variants.push(fill(buf[0],buf.length));
 for(let p=1;p<=Math.min(256,Math.floor(buf.length/2));p++){let ok=true;for(let i=p;i<buf.length;i++)if(buf[i]!==buf[i%p]){ok=false;break;}if(ok){variants.push(periodic(buf.subarray(0,p),buf.length));break;}}
 const counts=new Uint32Array(256);for(const b of buf)counts[b]++;let byte=0;for(let i=1;i<256;i++)if(counts[i]>counts[byte])byte=i;
 const changes=[];for(let i=0;i<buf.length;){if(buf[i]===byte){i++;continue;}let j=i;while(j<buf.length&&buf[j]!==byte)j++;changes.push({at:i,data:buf.subarray(i,j)});i=j;if(changes.length>4096)break;}
 if(changes.length<=4096&&changes.length>0)variants.push(patches(fill(byte,buf.length),changes));
 if(buf.length>=32&&buf.length%16===0){let seed=0n;for(let i=0;i<16;i++)seed|=BigInt(buf[i])<<(8n*BigInt(i));let v=seed,ok=true;for(let i=0;i<buf.length&&ok;i+=16){for(let j=0;j<16;j++)if(buf[i+j]!==Number((v>>(8n*BigInt(j)))&255n)){ok=false;break;}v=mulx(v);}if(ok)variants.push(gf128(seed,buf.length));}
 variants=variants.filter(v=>{try{return validDescriptor(v)&&read(v,0,buf.length).equals(buf);}catch{return false;}});
 variants.sort((a,b)=>Buffer.byteLength(JSON.stringify(a))-Buffer.byteLength(JSON.stringify(b)));
 return variants[0];
}
// Exact 16-byte-cell XOR. Algebraic for GF128; bounded reading for others.
function xorRawBytes(b){if(b.length%16)throw Error('Longueur doit etre multiple de 16');const r=Buffer.alloc(16);for(let i=0;i<b.length;i++)r[i%16]^=b[i];return r;}
function bufferToBigLE(b){let v=0n;for(let i=0;i<16;i++)v|=BigInt(b[i])<<(8n*BigInt(i));return v;}
function xorGF(d){const cnt=size(d)/16n;return mul(fieldValue(d.seed),geometrical(2n,cnt).s);}
function xorAll(d){const n=size(d);if(n%16n!==0n)throw Error('XOR global exige un multiple de 16 octets');
 if(d.kind==='gf128')return xorGF(d);
 if(d.kind==='fill')return n/16n%2n===0n?0n:bufferToBigLE(Buffer.alloc(16,d.byte));
 if(d.kind==='periodic'){
  const p=Buffer.from(d.pattern,'base64');const periodLen=16n*BigInt(p.length)/gcd(BigInt(p.length),16n);
  const cycleBytes=Number(periodLen);if(cycleBytes>MAX_READ)throw Error('Periode trop longue pour reduction');
  const whole=n/periodLen,rem=Number(n%periodLen);let a=0n;
  if(whole&1n)a^=bufferToBigLE(xorRawBytes(read(d,0,cycleBytes)));
  if(rem)a^=bufferToBigLE(xorRawBytes(read(d,0,rem)));return a;}
 if(d.kind==='literal'||d.kind==='lorenz96'){if(n>BigInt(MAX_READ))throw Error('Etat >1MiB');return bufferToBigLE(xorRawBytes(read(d,0,Number(n))));}
 if(d.kind==='patches'){
  let total=xorAll(d.base);for(const c of d.changes){const b=Buffer.from(c.data,'base64');const old=read(d.base,c.at,b.length);const pos=Number(nat(c.at)%16n);for(let i=0;i<b.length;i++){const shift=BigInt(((pos+i)%16)*8);total^=BigInt(b[i]^old[i])<<shift;}}return total;}
 throw Error('Reducteur inconnu');
}
function gcd(a,b){while(b){const r=a%b;a=b;b=r;}return a;}
function xorResiduesGF(d,k){if(d.kind!=='gf128')throw Error('Classes seulement pour GF128');if(!Number.isInteger(k)||k<1||k>256)throw Error('Paliers 1..256');const cells=size(d)/16n;let sum=0n;const step=pow(2n,BigInt(k));let value=fieldValue(d.seed);const parts=[];for(let r=0;r<k;r++){
  const rr=BigInt(r),count=rr<cells?1n+(cells-1n-rr)/BigInt(k):0n;
  const v=count?mul(value,geometrical(step,count).s):0n;
  sum^=v;parts.push({residue:r,cells:String(count),xor:hex(v)});value=mulx(value);
 }return {xor:hex(sum),classes:k,parts};
}
function compareGFParity(d){const v=xorResiduesGF(d,2);return {even:v.parts[0].xor,odd:v.parts[1].xor,total:v.xor};}
module.exports={QNE,sha,hex,nat,mul,mulx,pow,geometrical,gf128,lorenz96,literal,periodic,fill,patches,size,validDescriptor,read,discover,xorAll,xorResiduesGF,compareGFParity,bufferToBigLE};
