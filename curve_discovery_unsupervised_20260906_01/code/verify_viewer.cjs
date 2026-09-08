// Executes the actual page's control script with a small DOM/Plotly test double.
// This checks data and filtering logic; it is explicitly not a real browser test.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(__dirname,'..'),page=fs.readFileSync(path.join(root,'index.html'),'utf8');
const code=page.slice(page.lastIndexOf('<script>')+8,page.lastIndexOf('</script>'));
class Select {
  constructor(values=[]){this.options=values.map(value=>({value}));this.selectedIndex=0;this.textContent='';}
  set innerHTML(s){this.options=[...s.matchAll(/<option value="([^"]*)">/g)].map(x=>({value:x[1]}));this.selectedIndex=0;}
  get value(){return this.options[this.selectedIndex]?.value||'';}
  set value(v){this.selectedIndex=this.options.findIndex(o=>o.value===v);}
}
const el={group:new Select(),curve:new Select(),axis:new Select(['relative','hours','original']),norm:new Select(['minmax','maxabs','y']),prev:{},next:{},detail:{}};
let traces,layout;
const sandbox={document:{getElementById:id=>el[id]},Plotly:{react:(id,t,l)=>{traces=t;layout=l;}},console};
vm.createContext(sandbox);vm.runInContext(code,sandbox);
assert.equal(traces.length,218);assert.equal(traces.reduce((s,t)=>s+t.x.length,0),9674);
assert(traces.every(t=>t.line.simplify===false));
el.group.value='1';el.group.onchange();assert.equal(traces.length,46);
el.group.value='0';el.group.onchange();el.curve.value='C096';el.curve.onchange();assert.equal(traces.at(-1).name,'C096 / C2');
el.norm.value='maxabs';el.norm.onchange();assert(Math.min(...traces.at(-1).y)>.96);
el.axis.value='hours';el.axis.onchange();assert.equal(traces.length,159);
assert(Math.abs(traces.at(-1).x.at(-1)-100.703993)<.00001);
const expectedNext=el.curve.options[el.curve.selectedIndex+1].value;
el.next.onclick();assert(el.curve.value===expectedNext && el.curve.value!=='C096');
const result={script_syntax:true,all_218_curves_and_9674_points:true,cluster_filter:true,curve_highlight:true,normalization_switch:true,hour_filter_159:true,navigation:true,real_browser_test:false};
fs.writeFileSync(path.join(root,'experiments/viewer_logic_verification.json'),JSON.stringify(result,null,2));console.log(result);
