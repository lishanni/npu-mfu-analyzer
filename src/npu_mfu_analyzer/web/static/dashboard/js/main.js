// Main dashboard logic
const Dashboard = {
    data: null,
    currentHeatmapType: 'bandwidth',
    currentTransportFilter: 'all',
    currentStatusFilter: 'all',

    init(data) {
        this.data = data;
        this.renderMetrics();
        this.renderHeatmap();
        this.renderAnomalies();
        this.bindEvents();
        this.renderCharts();
    },

    renderMetrics() {
        const summary = this.data.summary;
        const grid = document.getElementById('metrics-grid');

        // Update header info
        document.getElementById('world-size').textContent = summary.world_size;
        document.getElementById('num-machines').textContent = Math.ceil(summary.world_size / 8);
        document.getElementById('npus-per-machine').textContent = '8';

        // Define metrics
        const metrics = [
            {
                label: '总通信量',
                value: this.formatNumber(summary.total_comm_data_mb),
                unit: 'MB',
                type: 'primary'
            },
            {
                label: '总通信时间',
                value: this.formatNumber(summary.total_comm_time_ms),
                unit: 'ms',
                type: 'primary'
            },
            {
                label: '平均带宽',
                value: this.formatNumber(summary.avg_bandwidth_gbps),
                unit: 'GB/s',
                type: 'primary'
            },
            {
                label: '峰值带宽',
                value: this.formatNumber(summary.peak_bandwidth_gbps),
                unit: 'GB/s',
                type: 'success'
            },
            {
                label: '节点内比例',
                value: (summary.intra_node_ratio * 100).toFixed(1),
                unit: '%',
                type: 'success'
            },
            {
                label: '节点间比例',
                value: (summary.inter_node_ratio * 100).toFixed(1),
                unit: '%',
                type: 'warning'
            },
            {
                label: '慢链路数',
                value: summary.slow_link_count,
                unit: '条',
                type: summary.slow_link_count > 0 ? 'danger' : 'success'
            },
            {
                label: '瓶颈链路数',
                value: summary.bottleneck_link_count,
                unit: '条',
                type: summary.bottleneck_link_count > 0 ? 'warning' : 'success'
            },
            {
                label: '通信效率评分',
                value: summary.efficiency_score.toFixed(1),
                unit: '/100',
                type: summary.efficiency_score >= 70 ? 'success' : summary.efficiency_score >= 50 ? 'warning' : 'danger'
            }
        ];

        grid.innerHTML = metrics.map(m => `
            <div class="metric-card ${m.type}">
                <div class="metric-value">${m.value}</div>
                <div class="metric-label">${m.label}</div>
                <div class="metric-unit">${m.unit}</div>
            </div>
        `).join('');
    },

    renderHeatmap() {
        const container = document.getElementById('heatmap-container');
        const heatmap = this.data.heatmap;
        const worldSize = heatmap.world_size;

        let matrix;
        if (this.currentHeatmapType === 'bandwidth') {
            matrix = heatmap.bandwidth_matrix;
        } else {
            matrix = heatmap.utilization_matrix;
        }

        // Find max value for color scaling
        let maxValue = 0;
        for (let i = 0; i < worldSize; i++) {
            for (let j = 0; j < worldSize; j++) {
                if (i !== j && matrix[i][j] > maxValue) {
                    maxValue = matrix[i][j];
                }
            }
        }

        // Build heatmap HTML
        let html = '<div class="heatmap" style="grid-template-columns: 50px repeat(' + worldSize + ', 45px);">';

        // Header row
        html += '<div class="heatmap-label"></div>';
        for (let j = 0; j < worldSize; j++) {
            html += '<div class="heatmap-label">R' + j + '</div>';
        }

        // Data rows
        for (let i = 0; i < worldSize; i++) {
            html += '<div class="heatmap-label">R' + i + '</div>';
            for (let j = 0; j < worldSize; j++) {
                if (i === j) {
                    html += '<div class="heatmap-cell diagonal">-</div>';
                } else {
                    const value = matrix[i][j];
                    const link = this.findLink(i, j);

                    // Apply filters
                    if (this.currentTransportFilter !== 'all' && link && link.transport_type !== this.currentTransportFilter) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }
                    if (this.currentStatusFilter === 'slow' && link && !link.is_slow_link) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }
                    if (this.currentStatusFilter === 'bottleneck' && link && !link.is_bottleneck) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }

                    let color, textColor;
                    if (link) {
                        if (link.is_slow_link) {
                            color = '#F44336';
                            textColor = 'white';
                        } else if (link.is_bottleneck) {
                            color = '#E91E63';
                            textColor = 'white';
                        } else {
                            const ratio = maxValue > 0 ? value / maxValue : 0;
                            color = this.getColorForRatio(ratio);
                            textColor = ratio > 0.5 ? 'white' : '#333';
                        }

                        const displayValue = this.currentHeatmapType === 'bandwidth'
                            ? value.toFixed(1)
                            : (value * 100).toFixed(0) + '%';

                        const tooltip = this.buildTooltip(link, value);
                        html += '<div class="heatmap-cell" style="background: ' + color + '; color: ' + textColor + '" ' +
                            'data-tooltip="' + tooltip + '">' + displayValue + '</div>';
                    } else {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                    }
                }
            }
        }

        html += '</div>';
        container.innerHTML = html;

        // Bind tooltip events
        this.bindTooltipEvents();
    },

    findLink(src, dst) {
        return this.data.links.find(l =>
            (l.src_rank === src && l.dst_rank === dst) ||
            (l.src_rank === dst && l.dst_rank === src)
        );
    },

    buildTooltip(link, value) {
        const parts = [
            'Rank ' + link.src_rank + ' ↔ ' + link.dst_rank,
            '类型: ' + link.transport_type.toUpperCase(),
            '带宽: ' + link.achieved_bandwidth_gbps + ' GB/s',
            '利用率: ' + (link.bandwidth_utilization * 100).toFixed(1) + '%',
            '数据量: ' + link.total_transit_size_mb + ' MB'
        ];
        if (link.is_slow_link) parts.push('⚠️ 慢链路');
        if (link.is_bottleneck) parts.push('🔥 瓶颈链路');
        return parts.join(' | ');
    },

    bindTooltipEvents() {
        const tooltip = document.getElementById('tooltip');
        document.querySelectorAll('.heatmap-cell:not(.diagonal)').forEach(cell => {
            cell.addEventListener('mouseenter', (e) => {
                const content = e.target.dataset.tooltip;
                if (content) {
                    tooltip.textContent = content;
                    tooltip.classList.add('visible');
                }
            });
            cell.addEventListener('mousemove', (e) => {
                tooltip.style.left = (e.pageX + 10) + 'px';
                tooltip.style.top = (e.pageY + 10) + 'px';
            });
            cell.addEventListener('mouseleave', () => {
                tooltip.classList.remove('visible');
            });
        });
    },

    getColorForRatio(ratio) {
        if (ratio < 0.3) {
            return 'rgb(76, 175, 80)';
        } else if (ratio < 0.6) {
            const r = Math.round(76 + (255 - 76) * (ratio - 0.3) / 0.3);
            const g = Math.round(175 + (193 - 175) * (ratio - 0.3) / 0.3);
            const b = Math.round(80 + (7 - 80) * (ratio - 0.3) / 0.3);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            const r = 255;
            const g = Math.max(0, Math.round(193 - 193 * (ratio - 0.6) / 0.4));
            const b = Math.max(0, Math.round(7 - 7 * (ratio - 0.6) / 0.4));
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }
    },

    renderAnomalies() {
        const anomalies = this.data.anomalies;

        // Slow links
        const slowTab = document.getElementById('slow-links-tab');
        if (anomalies.slow_links.length > 0) {
            slowTab.innerHTML = `
                <div class="alert alert-warning">
                    <h4>⚠️ 检测到 ${anomalies.slow_links.length} 条慢链路</h4>
                    <p>以下链路的带宽利用率显著低于平均值，可能影响训练性能。</p>
                </div>
                <table class="anomaly-table">
                    <thead>
                        <tr>
                            <th>链路</th>
                            <th>带宽</th>
                            <th>利用率</th>
                            <th>异常评分</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${anomalies.slow_links.map(link => `
                            <tr>
                                <td>R${link.src_rank} ↔ R${link.dst_rank}</td>
                                <td>${link.bandwidth_gbps} GB/s</td>
                                <td>${(link.utilization * 100).toFixed(1)}%</td>
                                <td><span class="badge slow">${(link.anomaly_score * 100).toFixed(1)}%</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            slowTab.innerHTML = `
                <div class="alert alert-info">
                    <h4>✓ 未检测到慢链路</h4>
                    <p>所有链路带宽利用率正常。</p>
                </div>
            `;
        }

        // Bottleneck links
        const bottleneckTab = document.getElementById('bottleneck-links-tab');
        if (anomalies.bottleneck_links.length > 0) {
            bottleneckTab.innerHTML = `
                <div class="alert alert-warning">
                    <h4>🔥 检测到 ${anomalies.bottleneck_links.length} 条瓶颈链路</h4>
                    <p>以下链路带宽利用率接近上限，可能成为通信瓶颈。</p>
                </div>
                <table class="anomaly-table">
                    <thead>
                        <tr>
                            <th>链路</th>
                            <th>带宽</th>
                            <th>利用率</th>
                            <th>数据量</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${anomalies.bottleneck_links.map(link => `
                            <tr>
                                <td>R${link.src_rank} ↔ R${link.dst_rank}</td>
                                <td>${link.bandwidth_gbps} GB/s</td>
                                <td><span class="badge bottleneck">${(link.utilization * 100).toFixed(1)}%</span></td>
                                <td>${link.total_size_mb} MB</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        } else {
            bottleneckTab.innerHTML = `
                <div class="alert alert-info">
                    <h4>✓ 未检测到瓶颈链路</h4>
                    <p>所有链路带宽利用率健康。</p>
                </div>
            `;
        }
    },

    bindEvents() {
        // Heatmap type change
        document.getElementById('heatmap-type').addEventListener('change', (e) => {
            this.currentHeatmapType = e.target.value;
            this.renderHeatmap();
        });

        // Transport filter change
        document.getElementById('transport-filter').addEventListener('change', (e) => {
            this.currentTransportFilter = e.target.value;
            this.renderHeatmap();
        });

        // Status filter change
        document.getElementById('status-filter').addEventListener('change', (e) => {
            this.currentStatusFilter = e.target.value;
            this.renderHeatmap();
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                e.target.classList.add('active');
                document.getElementById(e.target.dataset.tab + '-links-tab').classList.add('active');
            });
        });
    },

    renderCharts() {
        // Render bandwidth chart
        this.renderBarChart('bandwidth-chart', this.data.distributions.bandwidth_histogram, '带宽 (GB/s)');

        // Render utilization chart
        this.renderBarChart('utilization-chart', this.data.distributions.utilization_distribution, '利用率 (%)');
    },

    renderBarChart(canvasId, data, valueLabel) {
        const canvas = document.getElementById(canvasId);
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

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toFixed(1);
    }
};
