let priceChart = null;
let errorChart = null;
let timeChart = null;

function destroyChart(chart) {
  if (chart) {
    chart.destroy();
  }
  return null;
}

function baseOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function updatePageCharts(data) {
  if (!data || !data.charts) {
    return;
  }

  const priceCanvas = document.getElementById('chartPrices');
  const errorCanvas = document.getElementById('chartErrors');
  const timeCanvas = document.getElementById('chartTimes');

  const prices = data.charts.prices || {};
  const errors = data.charts.errors || {};
  const times = data.charts.times || {};

  if (priceCanvas) {
    priceChart = destroyChart(priceChart);

    priceChart = new Chart(priceCanvas, {
      type: 'bar',
      data: {
        labels: safeArray(prices.labels),
        datasets: [{
          label: 'Prix',
          data: safeArray(prices.values),
          backgroundColor: ['#2563eb', '#16a34a', '#f97316', '#7c3aed']
        }]
      },
      options: baseOptions()
    });
  }

  if (errorCanvas) {
    errorChart = destroyChart(errorChart);

    errorChart = new Chart(errorCanvas, {
      type: 'bar',
      data: {
        labels: safeArray(errors.labels),
        datasets: [{
          label: 'Erreur relative (%)',
          data: safeArray(errors.values),
          backgroundColor: ['#16a34a', '#f97316', '#7c3aed']
        }]
      },
      options: baseOptions()
    });
  }

  if (timeCanvas) {
    timeChart = destroyChart(timeChart);

    const safeTimeValues = safeArray(times.values).map(function (v) {
      return Math.max(Number(v) || 0.000001, 0.000001);
    });

    timeChart = new Chart(timeCanvas, {
      type: 'bar',
      data: {
        labels: safeArray(times.labels),
        datasets: [{
          label: 'Temps de calcul',
          data: safeTimeValues,
          backgroundColor: ['#2563eb', '#16a34a', '#f97316', '#7c3aed']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          y: {
            type: 'logarithmic',
            min: 0.000001,
            ticks: {
              callback: function (value) {
                return value + 's';
              }
            }
          }
        }
      }
    });
  }
}
