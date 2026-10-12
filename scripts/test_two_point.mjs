import assert from 'node:assert/strict';
import {
  correlationCurves,
  elementWidths,
  exactCorrelation,
  intervalLayout,
  pairStatistics,
  parseTwoPointState,
  separationAveragedJointMass,
  serializeTwoPointState,
  twoPointCsv,
  defaultTwoPointState,
} from '../app/js/two-point.js';

const close=(actual,expected,tol=1e-10)=>assert.ok(Math.abs(actual-expected)<=tol,`${actual} != ${expected} ± ${tol}`);

for(const s of [0,.03,.17,.25,.5]) {
  close(exactCorrelation(0,'A','A',s),0);
  close(exactCorrelation(0,'B','B',s),0);
}
assert.deepEqual(elementWidths(0),[.25,.25,.25,.25]);
console.log('Two-point uniform limit: ξ(s)=0 for both arrangements.');

close(exactCorrelation(1,'A','A',0),29/96);
close(exactCorrelation(1,'B','B',0),29/96);
close(exactCorrelation(1,'A','A',.25),-79/2304);
close(exactCorrelation(1,'B','B',.25),-169/2304);
const aa=pairStatistics(1,'A','A',.25);
const rowMass=aa.normalizedPairMass.map(row=>row.reduce((sum,value)=>sum+value,0));
close(rowMass[0],24/89);
close(aa.normalizedPairMass.flat().reduce((sum,value)=>sum+value,0),1);
for(const arrangement of ['A','B']) {
  const stats=pairStatistics(1,arrangement,arrangement,.25), widths=elementWidths(1);
  for(let i=0;i<4;i++)close(stats.overlaps[i].reduce((sum,value)=>sum+value,0),widths[i]);
  for(let j=0;j<4;j++)close(stats.overlaps.reduce((sum,row)=>sum+row[j],0),widths[j]);
  close(stats.normalizedPairMass.flat().reduce((sum,value)=>sum+value,0),1);
}
assert.notEqual(exactCorrelation(1,'A','A',.25),exactCorrelation(1,'B','B',.25),'Permuting the interval order changes its autocorrelation.');
assert.deepEqual(elementWidths(1),[.1,.2,.3,.4],'The one-point widths do not depend on arrangement.');
assert.deepEqual(intervalLayout(1,'A').widths,intervalLayout(1,'B').widths);
console.log('Strong-contrast fixture: ξ_A(0)=ξ_B(0)=29/96; at s=.25, ξ_A=-79/2304 and ξ_B=-169/2304. Overlap row/column sums recover element widths.');

const samples=2048;
for(const arrangement of ['A','B']) {
  const joint=separationAveragedJointMass(1,arrangement,arrangement,samples);
  const rowMarginal=joint.map(row=>row.reduce((sum,value)=>sum+value,0));
  const colMarginal=joint.reduce((acc,row)=>acc.map((value,j)=>value+row[j]),[0,0,0,0]);
  for(const value of [...rowMarginal,...colMarginal])close(value,.25,2e-6);
}
for(const arrangement of ['A','B']) {
  let mean=0;
  for(let k=0;k<samples;k++)mean+=exactCorrelation(1,arrangement,arrangement,(k+.5)/samples)/samples;
  close(mean,0,2e-6);
}
console.log('Autocorrelation fixtures: separation-integrated joint marginals recover equal masses and ∫ξ(s)ds=0.');

const curves=correlationCurves(1,101);
assert.equal(curves.length,101);
close(curves[0].separation,0);close(curves.at(-1).separation,.5);
assert.ok(curves.some(point=>Math.abs(point.xiA-point.xiB)>1e-3));

const state={contrast:.73,separation:.215,arrangement:'B'};
const serialized=serializeTwoPointState(state,'chapter=05&keep=yes');
assert.equal(parseTwoPointState(serialized).contrast,state.contrast);
assert.equal(parseTwoPointState(serialized).separation,state.separation);
assert.equal(parseTwoPointState(serialized).arrangement,state.arrangement);
assert.equal(new URLSearchParams(serialized).get('chapter'),'05');
assert.equal(new URLSearchParams(serialized).get('keep'),'yes');
assert.deepEqual(parseTwoPointState('?tp=9&tps=-2&tpa=invalid'),{contrast:1,separation:0,arrangement:'A'});
assert.deepEqual(defaultTwoPointState(),{contrast:1,separation:.25,arrangement:'A'});
assert.deepEqual(parseTwoPointState(''),defaultTwoPointState());
const csv=twoPointCsv({contrast:1,arrangement:'A',separation:.25}), csvLines=csv.split('\n');
assert.equal(csvLines[0],'row_type,t,arrangement,s,element_i,element_j,width_i,width_j,density_i,density_j,volume_pair_weight,unnormalized_mass_pair_weight,normalized_mass_pair_weight,xi_A,xi_B');
const metadata=csvLines.find(line=>line.startsWith('metadata,')).split(',');
close(Number(metadata[13]),-79/2304);close(Number(metadata[14]),-169/2304);
const curve=csvLines.find(line=>line.startsWith('curve,1,,0.25,')).split(',');
close(Number(curve[13]),-79/2304);close(Number(curve[14]),-169/2304);
assert.equal(csvLines.filter(line=>line.startsWith('pair,')).length,16,'CSV includes all nonzero autocorrelation pair rows for both arrangements.');
console.log('Two-point URL state round-trips and preserves unrelated query parameters.');
