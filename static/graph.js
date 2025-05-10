document.addEventListener('DOMContentLoaded', function() {
    const chartCanvas = document.getElementById('resultsChart');
    if (chartCanvas) {
        const labels = JSON.parse(chartCanvas.dataset.labels);
        const data = JSON.parse(chartCanvas.dataset.data);

        const ctx = chartCanvas.getContext('2d');

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Level',
                    data: data,
                    backgroundColor: data.map(value => {
                        if (value >= 3) return 'rgba(255, 99, 132, 0.7)';       // High/Very High
                        if (value === 2) return 'rgba(255, 206, 86, 0.7)';      // Medium
                        return 'rgba(75, 192, 192, 0.7)';                       // Low/Very Low
                    }),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 4,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                switch (value) {
                                    case 0: return "Very Low";
                                    case 1: return "Low";
                                    case 2: return "Medium";
                                    case 3: return "High";
                                    case 4: return "Very High";
                                }
                            }
                        }
                    }
                }
            }
        });
    }
});
