// Heatmap component
const Heatmap = {
    colors: {
        hccs: '#4CAF50',
        rdma: '#2196F3',
        roce: '#9C27B0',
        slow: '#F44336',
        bottleneck: '#E91E63'
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

    findLink(links, src, dst) {
        return links.find(l =>
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

    render(matrix, links, options = {}) {
        const {
            type = 'bandwidth',
            transportFilter = 'all',
            statusFilter = 'all',
            container = document.getElementById('heatmap-container')
        } = options;

        if (!container) return;

        const worldSize = matrix.length;
        let dataMatrix = matrix;

        if (type === 'utilization') {
            dataMatrix = matrix.map(row => row.map(val => val / 100));
        }

        // Find max value for color scaling
        let maxValue = 0;
        for (let i = 0; i < worldSize; i++) {
            for (let j = 0; j < worldSize; j++) {
                if (i !== j && dataMatrix[i][j] > maxValue) {
                    maxValue = dataMatrix[i][j];
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
                    const value = dataMatrix[i][j];
                    const link = this.findLink(links, i, j);

                    // Apply filters
                    if (transportFilter !== 'all' && link && link.transport_type !== transportFilter) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }
                    if (statusFilter === 'slow' && link && !link.is_slow_link) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }
                    if (statusFilter === 'bottleneck' && link && !link.is_bottleneck) {
                        html += '<div class="heatmap-cell" style="background: #f5f5f5; color: #999;">-</div>';
                        continue;
                    }

                    let color, textColor;
                    if (link) {
                        if (link.is_slow_link) {
                            color = this.colors.slow;
                            textColor = 'white';
                        } else if (link.is_bottleneck) {
                            color = this.colors.bottleneck;
                            textColor = 'white';
                        } else {
                            const ratio = maxValue > 0 ? value / maxValue : 0;
                            color = this.getColorForRatio(ratio);
                            textColor = ratio > 0.5 ? 'white' : '#333';
                        }

                        const displayValue = type === 'bandwidth'
                            ? matrix[i][j].toFixed(1)
                            : (matrix[i][j] * 100).toFixed(0) + '%';

                        const tooltip = this.buildTooltip(link, matrix[i][j]);
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
    },

    bindTooltipEvents() {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;

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
    }
};