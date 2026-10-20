let eulerPaths=null, eulerSteps=null;
function destroy(c){ if(c) c.destroy(); }
function updatePageCharts(data){
  const paths=data.charts.euler_paths;
  if(document.getElementById('eulerPathsChart')){destroy(eulerPaths); const datasets=(paths.paths||[]).map((p,i)=>({label:'Traj '+(i+1),data:p,borderColor:['#2563eb','#16a34a','#f97316','#7c3aed','#0891b2','#dc2626','#475569','#d97706'][i%8],pointRadius:0,borderWidth:1.4,tension:.15})); eulerPaths=new Chart(document.getElementById('eulerPathsChart'),{type:'line',data:{labels:paths.time,datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}}})}
  const st=data.charts.euler_steps;
  if(document.getElementById('eulerStepChart')){destroy(eulerSteps); eulerSteps=new Chart(document.getElementById('eulerStepChart'),{type:'line',data:{labels:st.N,datasets:[{label:'Prix Euler',data:st.prices,borderColor:'#f97316',pointRadius:3,tension:.25},{label:'Biais',data:st.biases,borderColor:'#dc2626',pointRadius:3,tension:.25}]},options:{responsive:true,maintainAspectRatio:false}})}
}
