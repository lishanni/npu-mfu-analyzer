// Chart rendering components
const Charts = {
    renderBarChart(canvasId, data, valueLabel) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (!data || data.length === 0) {
            ctx.fillStyle = '#80868b';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('暂无数据', width / 2, height / 2);
            return;
        }

        // Find max value
        const maxValue = Math.max(...data.map(d => d.count));
        const barWidth = chartWidth / data.length;
        const barGap = 2;

        // Draw bars
        data.forEach((item, i) => {
            const barHeight = (item.count / maxValue) * chartHeight;
            const x = padding.left + i * barWidth;
            const y = padding.top + chartHeight - barHeight;

            // Draw bar
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, '#1a73e8');
            gradient.addColorStop(1, '#4285f4');
            ctx.fillStyle = gradient;
            ctx.fillRect(x + barGap, y, barWidth - barGap * 2, barHeight);

            // Draw count on top
            ctx.fillStyle = '#333';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(item.count, x + barWidth / 2, y - 5);
        });

        // Draw axes
        ctx.strokeStyle = '#dadce0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + chartHeight);
        ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
        ctx.stroke();

        // Draw Y axis labels
        ctx.fillStyle = '#5f6368';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const value = Math.round((maxValue / 5) * i);
            const y = padding.top + chartHeight - (chartHeight / 5) * i;
            ctx.fillText(value, padding.left - 8, y + 4);
        }

        // Draw X axis labels (show every nth label to avoid crowding)
        const labelStep = Math.ceil(data.length / 8);
        ctx.textAlign = 'center';
        data.forEach((item, i) => {
            if (i % labelStep === 0) {
                const x = padding.left + i * barWidth + barWidth / 2;
                ctx.fillText(item.label, x, padding.top + chartHeight + 16);
            }
        });
    },

    renderPieChart(canvasId, data, labelKey, valueKey) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(centerX, centerY) - 40;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (!data || data.length === 0) {
            ctx.fillStyle = '#80868b';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('暂无数据', centerX, centerY);
            return;
        }

        const colors = ['#1a73e8', '#34a853', '#fbbc04', '#ea4335', '#9c27b0', '#ff9800', '#00bcd4', '#795548'];
        const total = data.reduce((sum, item) => sum + item[valueKey], 0);

        let startAngle = -Math.PI / 2;

        data.forEach((item, i) => {
            const sliceAngle = (item[valueKey] / total) * 2 * Math.PI;

            // Draw slice
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
            ctx.closePath();

            ctx.fillStyle = colors[i % colors.length];
            ctx.fill();

            // Draw label
            const midAngle = startAngle + sliceAngle / 2;
            const labelX = centerX + Math.cos(midAngle) * (radius * 0.7);
            const labelY = centerY + Math.sin(midAngle) * (radius * 0.7);

            const percentage = ((item[valueKey] / total) * 100).toFixed(1);

            ctx.fillStyle = 'white';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            if (percentage > 5) {
                ctx.fillText(percentage + '%', labelX, labelY);
            }

            startAngle += sliceAngle;
        });

        // Draw legend
        const legendX = 20;
        let legendY = height - 20;
        data.forEach((item, i) => {
            ctx.fillStyle = colors[i % colors.length];
            ctx.fillRect(legendX, legendY - 8, 12, 12);

            ctx.fillStyle = '#333';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(item[labelKey], legendX + 16, legendY - 2);

            legendY -= 20;
        });
    },

    renderLineChart(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (!data || data.length === 0) {
            ctx.fillStyle = '#80868b';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('暂无数据', width / 2, height / 2);
            return;
        }

        // Find min/max values
        const values = data.map(d => d.value);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue || 1;

        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = '#1a73e8';
        ctx.lineWidth = 2;

        data.forEach((item, i) => {
            const x = padding.left + (i / (data.length - 1)) * chartWidth;
            const y = padding.top + chartHeight - ((item.value - minValue) / valueRange) * chartHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw points
        data.forEach((item, i) => {
            const x = padding.left + (i / (data.length - 1)) * chartWidth;
            const y = padding.top + chartHeight - ((item.value - minValue) / valueRange) * chartHeight;

            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#1a73e8';
            ctx.fill();
        });

        // Draw axes
        ctx.strokeStyle = '#dadce0';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + chartHeight);
        ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
        ctx.stroke();

        // Draw Y axis labels
        ctx.fillStyle = '#5f6368';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const value = minValue + (valueRange / 5) * i;
            const y = padding.top + chartHeight - (chartHeight / 5) * i;
            ctx.fillText(value.toFixed(1), padding.left - 8, y + 4);
        }

        // Draw X axis labels
        const labelStep = Math.ceil(data.length / 8);
        ctx.textAlign = 'center';
        data.forEach((item, i) => {
            if (i % labelStep === 0) {
                const x = padding.left + (i / (data.length - 1)) * chartWidth;
                ctx.fillText(item.label, x, padding.top + chartHeight + 16);
            }
        });
    }
};