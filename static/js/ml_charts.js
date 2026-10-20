let mlPriceChart=null;
function destroy(c){ if(c) c.destroy(); }
function updatePageCharts(data){
  if(document.getElementById('chartPrices')){destroy(mlPriceChart); mlPriceChart=new Chart(document.getElementById('chartPrices'),{type:'bar',data:{labels:data.charts.prices.labels,datasets:[{label:'Prix',data:data.charts.prices.values,backgroundColor:['#2563eb','#16a34a','#f97316','#7c3aed']}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}})}
}
