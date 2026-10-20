/* Même logique que le dashboard pour la page comparaison. */
let priceChart=null, errorChart=null, timeChart=null;
function destroy(c){ if(c) c.destroy(); }
function baseOptions(){ return {responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true}}}; }
function updatePageCharts(data){
  if(document.getElementById('chartPrices')){
    destroy(priceChart);
    priceChart = new Chart(document.getElementById('chartPrices'), {type:'bar', data:{labels:data.charts.prices.labels, datasets:[{label:'Prix', data:data.charts.prices.values, backgroundColor:['#2563eb','#16a34a','#f97316','#7c3aed']}]}, options:baseOptions()});
  }
  if(document.getElementById('chartErrors')){
    destroy(errorChart);
    errorChart = new Chart(document.getElementById('chartErrors'), {type:'bar', data:{labels:data.charts.errors.labels, datasets:[{label:'Erreur %', data:data.charts.errors.values, backgroundColor:['#16a34a','#f97316','#7c3aed']}]}, options:baseOptions()});
  }
  if(document.getElementById('chartTimes')){
    destroy(timeChart);
    timeChart = new Chart(document.getElementById('chartTimes'), {type:'bar', data:{labels:data.charts.times.labels, datasets:[{label:'Temps', data:data.charts.times.values, backgroundColor:['#2563eb','#16a34a','#f97316','#7c3aed']}]}, options:{...baseOptions(), scales:{y:{type:'logarithmic', min:0.000001}}}});
  }
}
