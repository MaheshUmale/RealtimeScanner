// static/js/charting.js

/**
 * A wrapper class for the Lightweight Charts library to simplify chart creation and management.
 */
class StockChart {
    /**
     * @param {string} containerId The ID of the HTML element where the chart will be rendered.
     */
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(this.containerId);
        this.chart = null;
    }

    /**
     * Clears any existing chart and creates a new candlestick chart with a volume histogram.
     * @param {Array<Object>} chartData - The data for the chart.
     */
    createCandlestickChartWithVolume(chartData) {
        // Destroy any existing chart instance before creating a new one
        this.destroy();

        if (!this.container || chartData.length === 0) {
            console.error("Chart container not found or no data provided.");
            return;
        }

        this.chart = LightweightCharts.createChart(this.container, {
            width: this.container.clientWidth,
            height: 400,
            layout: {
                background: { type: 'solid', color: '#ffffff' },
                textColor: '#333',
            },
            grid: {
                vertLines: { color: 'rgba(197, 203, 206, 0.5)' },
                horzLines: { color: 'rgba(197, 203, 206, 0.5)' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        // Candlestick Series
        const candleSeries = this.chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });

        // Volume Series
        const volumeSeries = this.chart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: '', // Set to an empty string to overlay on the main chart
        });
        
        // Set price scale for volume on the bottom
        this.chart.priceScale('').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        // Prepare and set data
        const candleData = chartData.map(d => ({
            time: d.time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
        }));

        const volumeData = chartData.map(d => ({
            time: d.time,
            value: d.volume,
            color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
        }));

        candleSeries.setData(candleData);
        volumeSeries.setData(volumeData);
        
        this.chart.timeScale().fitContent();
    }

    /**
     * Removes the chart from the DOM and cleans up resources.
     */
    destroy() {
        if (this.chart) {
            this.chart.remove();
            this.chart = null;
        }
    }
}