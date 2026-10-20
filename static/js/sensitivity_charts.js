let sensS=null, sensSigma=null;
function destroy(c){ if(c) c.destroy(); }
function updatePageCharts(data){
  fetch('/api/sensitivity',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(getParams())}).then(r=>r.json()).then(s=>{
    const rows=s.rows||[];
    const tbody=document.querySelector('#sensTable tbody');
    if(tbody) tbody.innerHTML=rows.slice(0,40).map(r=>`<tr><td>${r.variable||''}</td><td>${fmt(r.value,4)}</td><td>${fmt(r.price,4)}</td></tr>`).join('');
    const sRows=rows.filter(r=>String(r.variable).includes('S'));
    const sigRows=rows.filter(r=>String(r.variable).toLowerCase().includes('sigma'));
    if(document.getElementById('sensSChart')){destroy(sensS); sensS=new Chart(document.getElementById('sensSChart'),{type:'line',data:{labels:sRows.map(r=>fmt(r.value,2)),datasets:[{label:'Prix',data:sRows.map(r=>r.price),borderColor:'#2563eb',pointRadius:2}]},options:{responsive:true,maintainAspectRatio:false}})}
    if(document.getElementById('sensSigmaChart')){destroy(sensSigma); sensSigma=new Chart(document.getElementById('sensSigmaChart'),{type:'line',data:{labels:sigRows.map(r=>fmt(r.value,2)),datasets:[{label:'Prix',data:sigRows.map(r=>r.price),borderColor:'#7c3aed',pointRadius:2}]},options:{responsive:true,maintainAspectRatio:false}})}
  });
}
