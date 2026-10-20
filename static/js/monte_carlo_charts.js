let mcTerminal=null, mcPayoff=null, mcConv=null;
function destroy(c){ if(c) c.destroy(); }
function updatePageCharts(data){
  const h1=data.charts.mc_terminal_hist, h2=data.charts.mc_payoff_hist, conv=data.charts.mc_convergence;
  if(document.getElementById('mcTerminalChart')){destroy(mcTerminal); mcTerminal=new Chart(document.getElementById('mcTerminalChart'),{type:'bar',data:{labels:h1.labels,datasets:[{label:'S_T',data:h1.values,backgroundColor:'#2563eb'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}}})}
  if(document.getElementById('mcPayoffChart')){destroy(mcPayoff); mcPayoff=new Chart(document.getElementById('mcPayoffChart'),{type:'bar',data:{labels:h2.labels,datasets:[{label:'Payoffs',data:h2.values,backgroundColor:'#16a34a'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}}})}
  if(document.getElementById('mcConvChart')){destroy(mcConv); mcConv=new Chart(document.getElementById('mcConvChart'),{type:'line',data:{labels:conv.M,datasets:[{label:'Monte-Carlo',data:conv.prices,borderColor:'#16a34a',pointRadius:2,tension:.25},{label:'Black-Scholes',data:conv.M.map(()=>conv.bs),borderColor:'#dc2626',borderDash:[6,4],pointRadius:0}]},options:{responsive:true,maintainAspectRatio:false}})}
}
