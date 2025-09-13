document.addEventListener("DOMContentLoaded", () => {
  if (!chartData || chartData.length === 0) {
    console.warn("No data available for chart.");
    return;
  }

  const ctx = document.getElementById("salinityChart").getContext("2d");

  // Destroy old chart if it exists (avoid multiple charts on re-render)
  if (window.myChart) {
    window.myChart.destroy();
  }

  window.myChart = new Chart(ctx, {
    type: "line", // You can change to 'bar', 'pie', etc.
    data: {
      labels: chartData.years || [],
      datasets: [
        {
          label: `${region} - ${parameter}`,
          data: chartData.values || [], // 👈 we’ll fix this in app.js
          borderColor: "rgba(54, 162, 235, 1)",
          backgroundColor: "rgba(54, 162, 235, 0.2)",
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: `${parameter.toUpperCase()} trends in ${region}`,
        },
      },
      scales: {
        y: {
          beginAtZero: false,
        },
      },
    },
  });
});

