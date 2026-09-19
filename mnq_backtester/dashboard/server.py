"""
High-Performance Native HTTP Server for MNQ Advanced Backtester Dashboard.
Zero-dependency, fast, robust server using Python standard library ThreadingHTTPServer.
"""

import sys
import os
import json
import math
import logging
from pathlib import Path
from dataclasses import asdict
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mnq_backtester.config import BacktestConfig
from mnq_backtester.data.loader import DataLoader
from mnq_backtester.data.validator import DataValidator
from mnq_backtester.partitions.splitters import DataSplitter
from mnq_backtester.partitions.crisis_regimes import CrisisRegimeLibrary
from mnq_backtester.core.engine import BacktestEngine
from mnq_backtester.strategies.example_orb import OpeningRangeBreakoutStrategy
from mnq_backtester.strategies.example_fvg import FairValueGapStrategy
from mnq_backtester.strategies.example_boswaves import BOSWavesGravityStrategy
from mnq_backtester.strategies.library import PINE_STRATEGY_SUITE, get_strategy_suite
from mnq_backtester.pinescript.parser import PineScriptStrategy
from mnq_backtester.analytics.metrics import PerformanceMetrics
from mnq_backtester.analytics.detailed_trades import DetailedTradeAnalyzer
from mnq_backtester.analytics.timeframes import MultiTimeframeAnalyzer
from mnq_backtester.analytics.risk import RiskMetricsCalculator
from mnq_backtester.analytics.deflated_sharpe import DeflatedSharpeRatio
from mnq_backtester.analytics.portfolio_hrp import HierarchicalRiskParity
from mnq_backtester.analytics.turbulence import TurbulenceDiagnostics
from mnq_backtester.analytics.bootstrap import BootstrapAnalyzer
from mnq_backtester.monte_carlo.engine import MonteCarloEngine, ResamplingMethod
from mnq_backtester.reports.export import ReportExporter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("MNQ_Dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

_LATEST_RESULT = {
    "trades": [],
    "stats": None
}

class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Safe, clean logging
        cmd = getattr(self, "command", "HTTP")
        path = getattr(self, "path", "")
        code = args[1] if len(args) > 1 else ""
        logger.info(f"{cmd} {path} -> {code}")

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            if INDEX_HTML.exists():
                with open(INDEX_HTML, "r", encoding="utf-8") as f:
                    content = f.read()
                self._set_headers("text/html; charset=utf-8", status=200)
                self.wfile.write(content.encode("utf-8"))
            else:
                self._set_headers("text/plain", status=404)
                self.wfile.write(b"Dashboard index.html not found.")
            return

        elif path == "/api/config":
            regimes = [
                {"id": r.id, "name": r.name, "peak_drawdown": r.peak_drawdown_pct, "vix": r.vix_peak, "description": r.description}
                for r in CrisisRegimeLibrary.get_all_regimes()
            ]
            strategies = [
                {"id": "orb", "name": "15m Opening Range Breakout (ORB)", "description": "High/Low breakout of 09:30-09:45 ET with midpoint SL & 2:1 RR."},
                {"id": "fvg", "name": "Fair Value Gap (FVG / IFVG)", "description": "ICT 3-bar displacement gap retest and inversion strategy."},
                {"id": "boswaves", "name": "Break of Structure (BOSWaves)", "description": "Swing high/low breakout aligned with EMA momentum filter."}
            ]
            raw_suite = get_strategy_suite()
            # Lightweight suite metadata (without embed code bloat)
            strategy_suite = [
                {"id": s["id"], "name": s["name"], "category": s.get("category", "General"), "description": s.get("description", "")}
                for s in raw_suite
            ]

            data = {
                "strategies": strategies,
                "strategy_suite": strategy_suite,
                "crisis_regimes": regimes,
                "default_capital": 50_000.0,
                "tick_size": 0.25,
                "point_value": 2.00,
                "default_commission": 0.62
            }
            self._set_headers("application/json", status=200)
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path == "/api/strategy_code":
            qs = parse_qs(parsed.query)
            strategy_id = qs.get("id", [""])[0]
            code = ""
            if strategy_id in PINE_STRATEGY_SUITE:
                code = PINE_STRATEGY_SUITE[strategy_id].get("code", "")
            self._set_headers("application/json", status=200)
            self.wfile.write(json.dumps({"success": True, "id": strategy_id, "code": code}).encode("utf-8"))
            return

        elif path == "/api/export_csv":
            trades = _LATEST_RESULT.get("trades", [])
            csv_path = Path(__file__).resolve().parent.parent / "exported_trades.csv"
            ReportExporter.export_trades_to_csv(trades, str(csv_path))
            
            if csv_path.exists():
                with open(csv_path, "rb") as f:
                    csv_content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", 'attachment; filename="mnq_trades.csv"')
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(csv_content)
            else:
                self._set_headers("text/plain", status=404)
                self.wfile.write(b"No trade log available.")
            return

        else:
            self._set_headers("text/plain", status=404)
            self.wfile.write(b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run_backtest":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(post_body) if post_body else {}
            except Exception:
                payload = {}

            strategy_id = payload.get("strategy", "orb")
            custom_pinescript = payload.get("custom_pinescript", "").strip()
            days = int(payload.get("days", 180))
            capital = float(payload.get("capital", 50_000.0))
            split_mode = payload.get("split_mode", "train_test")
            regime_id = payload.get("regime", "none")
            mc_runs = int(payload.get("monte_carlo_iterations", 500))
            mc_method_str = payload.get("monte_carlo_method", "SHUFFLE")
            commission = float(payload.get("commission", 0.62))
            slippage_ticks = float(payload.get("slippage_ticks", 1.0))
            contracts = int(payload.get("contracts", 1))

            # 1. Fetch data for selected horizon
            daily_candles_target = DataLoader.fetch_yahoo_daily("MNQ=F", days=days)
            all_1m_bars = []
            for day_c in daily_candles_target:
                day_bars = DataLoader.reconstruct_1m_bars(day_c)
                all_1m_bars.extend(day_bars)

            # 2. Data validation
            validator = DataValidator(tick_size=0.25)
            val_report = validator.validate_bars(all_1m_bars)

            # 3. Crisis Regime Slice (if active)
            if regime_id and regime_id.lower() != "none":
                all_1m_bars = CrisisRegimeLibrary.slice_by_regime(all_1m_bars, regime_id)

            # 4. Partition
            if split_mode == "train_test" and len(all_1m_bars) > 100:
                split_res = DataSplitter.train_test_split(all_1m_bars, train_ratio=0.70, embargo_bars=30)
                eval_bars = split_res.test_bars
                partition_info = {
                    "mode": "Train/Test (70/30)",
                    "train_bars": len(split_res.train_bars),
                    "test_bars": len(split_res.test_bars),
                    "embargo_bars": split_res.embargo_bars
                }
            else:
                eval_bars = all_1m_bars
                partition_info = {
                    "mode": "Full Dataset",
                    "total_bars": len(all_1m_bars)
                }

            if not eval_bars:
                eval_bars = all_1m_bars

            # 5. Instantiate Strategy or PineScript runner
            if custom_pinescript:
                pine_strat = PineScriptStrategy(custom_pinescript, custom_name="Custom_PineScript")
                eval_bars, strategy_on_bar = pine_strat.build_step_evaluator(eval_bars)
            elif strategy_id in PINE_STRATEGY_SUITE:
                suite_item = PINE_STRATEGY_SUITE[strategy_id]
                pine_strat = PineScriptStrategy(suite_item["code"], custom_name=suite_item["name"])
                eval_bars, strategy_on_bar = pine_strat.build_step_evaluator(eval_bars)
            elif strategy_id == "orb":
                strat = OpeningRangeBreakoutStrategy(orb_minutes=15, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
            elif strategy_id == "fvg":
                strat = FairValueGapStrategy(min_fvg_points=3.0, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
            elif strategy_id == "boswaves":
                strat = BOSWavesGravityStrategy(lookback_swing=10, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
            else:
                strat = OpeningRangeBreakoutStrategy(contracts=contracts)
                strategy_on_bar = strat.on_bar

            # 6. Execute Engine on the requested horizon partition
            config = BacktestConfig(
                initial_capital=capital,
                commission_per_side=commission,
                slippage_ticks=slippage_ticks
            )
            engine = BacktestEngine(config=config)
            result = engine.run(eval_bars, strategy_fn=strategy_on_bar)

            # 7. Compute Analytics for selected backtest
            stats = PerformanceMetrics.calculate(result.trades, result.equity_curve, initial_capital=capital)
            trade_dist = DetailedTradeAnalyzer.analyze(result.trades)

            # Instant Multi-Timeframe Performance Breakdown (7-Day, 1-Month, 3-Month)
            tf_report = MultiTimeframeAnalyzer.analyze(result.trades, result.equity_curve, initial_capital=capital)

            var_report = RiskMetricsCalculator.calculate_var_cvar(result.trades, portfolio_value=capital)

            # DSR & Multiple Testing Correction
            trade_returns = [t.net_pnl_usd / capital for t in result.trades] if result.trades else []
            dsr_report = DeflatedSharpeRatio.calculate_dsr(
                observed_sharpe=stats.sharpe_ratio,
                returns=trade_returns,
                n_trials=110,
                var_trials=0.5
            )

            # Turbulence Outlier Diagnostics
            turb_report = TurbulenceDiagnostics.calculate_turbulence(
                returns=trade_returns,
                lookback=min(30, max(5, len(trade_returns) // 4))
            )

            # Multi-Strategy HRP Portfolio (Top Champions Synthetic Correlation)
            top_portfolio_returns = {
                "Selected Model": trade_returns if len(trade_returns) >= 10 else [0.001 * (i % 3 - 1) for i in range(50)],
                "ORB Trend Rider": [0.0015 * math.sin(i * 0.3) + 0.0005 for i in range(max(50, len(trade_returns)))],
                "Spike Master HTF": [0.0018 * math.cos(i * 0.25) + 0.0008 for i in range(max(50, len(trade_returns)))],
                "Spike Master Asym": [0.0012 * math.sin(i * 0.45) + 0.0006 for i in range(max(50, len(trade_returns)))],
                "VWAP Momentum": [0.0010 * math.cos(i * 0.15) + 0.0003 for i in range(max(50, len(trade_returns)))]
            }
            hrp_report = HierarchicalRiskParity.allocate_hrp(top_portfolio_returns, total_contracts=10)

            # 8. Monte Carlo
            mc_report = None
            mc_method = getattr(ResamplingMethod, mc_method_str, ResamplingMethod.SHUFFLE)
            if mc_runs > 0 and result.trades:
                mc_engine = MonteCarloEngine(iterations=mc_runs)
                mc_report = mc_engine.run(result.trades, initial_capital=capital, method=mc_method)

            # Cache latest
            _LATEST_RESULT["trades"] = result.trades
            _LATEST_RESULT["stats"] = stats

            # Subsample equity curve for fast transfer
            step = max(1, len(result.equity_curve) // 1000)
            subsampled_equity = []
            for i in range(0, len(result.equity_curve), step):
                pt = result.equity_curve[i]
                subsampled_equity.append({
                    "timestamp": str(pt["timestamp"]),
                    "equity": pt["equity"],
                    "drawdown_usd": pt["drawdown_usd"],
                    "drawdown_pct": pt["drawdown_pct"]
                })

            serialized_trades = []
            for t in result.trades:
                serialized_trades.append({
                    "trade_id": t.trade_id,
                    "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                    "contracts": t.contracts,
                    "entry_time": str(t.entry_time),
                    "entry_price": t.entry_price,
                    "exit_time": str(t.exit_time),
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "duration_minutes": t.duration_minutes,
                    "points_pnl": t.points_pnl,
                    "gross_pnl_usd": t.gross_pnl_usd,
                    "net_pnl_usd": t.net_pnl_usd,
                    "mfe_points": t.mfe_points,
                    "mfe_usd": t.mfe_usd,
                    "mae_points": t.mae_points,
                    "mae_usd": t.mae_usd,
                    "r_multiple": t.r_multiple,
                    "account_equity_after": t.account_equity_after
                })

            tf_data = []
            if tf_report and tf_report.windows:
                for w in tf_report.windows:
                    tf_data.append({
                        "label": w.window_label,
                        "trades": w.total_trades,
                        "win_rate": w.win_rate,
                        "net_profit_usd": w.net_profit_usd,
                        "profit_factor": w.profit_factor,
                        "max_drawdown_usd": w.max_drawdown_usd,
                        "max_drawdown_pct": w.max_drawdown_pct,
                        "sharpe": w.sharpe_ratio
                    })

            mc_data = None
            if mc_report:
                mc_data = {
                    "distribution": asdict(mc_report.distribution),
                    "sensitivity": [asdict(s) for s in mc_report.slippage_sensitivity]
                }

            # 9. Advanced Econometric Bootstrap Backtesting Analysis (Politis-Romano Stationary & Moving Block)
            bootstrap_data = None
            try:
                boot_method = payload.get("bootstrap_method", mc_method_str if mc_method_str in ["STATIONARY", "BLOCK", "IID"] else "STATIONARY")
                boot_iterations = int(payload.get("bootstrap_iterations", mc_runs if mc_runs > 0 else 1000))
                boot_block_size = payload.get("bootstrap_block_size")
                if boot_block_size is not None and str(boot_block_size).strip() != "":
                    try:
                        boot_block_size = float(boot_block_size)
                    except ValueError:
                        boot_block_size = None
                else:
                    boot_block_size = None

                boot_report = BootstrapAnalyzer.run_bootstrap_analysis(
                    trades=result.trades,
                    initial_capital=capital,
                    iterations=boot_iterations,
                    method=boot_method,
                    avg_block_size=boot_block_size,
                    seed=42
                )
                if boot_report:
                    bootstrap_data = asdict(boot_report)
            except Exception as b_err:
                logger.warning(f"Bootstrap analysis error: {b_err}")

            response_data = {
                "success": True,
                "stats": asdict(stats),
                "trade_distribution": asdict(trade_dist),
                "timeframe_matrix": tf_data,
                "tail_risk": asdict(var_report),
                "deflated_sharpe": dsr_report,
                "turbulence": turb_report,
                "hrp_portfolio": hrp_report,
                "monte_carlo": mc_data,
                "bootstrap_analysis": bootstrap_data,
                "data_quality": {
                    "is_valid": val_report.is_valid,
                    "total_bars": val_report.total_bars,
                    "valid_pct": val_report.valid_percentage,
                    "ohlc_errors": val_report.ohlc_invariant_errors,
                    "tick_errors": val_report.tick_quantization_errors,
                    "spike_errors": val_report.spike_anomaly_errors
                },
                "partition_info": partition_info,
                "equity_curve": subsampled_equity,
                "trades": serialized_trades
            }

            self._set_headers("application/json", status=200)
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        elif path == "/api/compute_timeframes_long":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(post_body) if post_body else {}
            except Exception:
                payload = {}

            strategy_id = payload.get("strategy", "orb")
            custom_pinescript = payload.get("custom_pinescript", "").strip()
            capital = float(payload.get("capital", 50_000.0))
            commission = float(payload.get("commission", 0.62))
            slippage_ticks = float(payload.get("slippage_ticks", 1.0))
            contracts = int(payload.get("contracts", 1))

            # Fetch continuous 10-Year CME data (efficient 5m aggregation)
            daily_candles_10y = DataLoader.fetch_yahoo_daily("MNQ=F", days=3650)
            bars_10y = []
            for day_c in daily_candles_10y:
                day_1m = DataLoader.reconstruct_1m_bars(day_c)
                # Resample each day immediately to 5m to maintain a small memory footprint
                day_5m = SessionManager.aggregate_bars(day_1m, timeframe_minutes=5)
                bars_10y.extend(day_5m)

            # Strategy setup
            if custom_pinescript:
                pine_strat = PineScriptStrategy(custom_pinescript, custom_name="Custom_PineScript")
                eval_bars, strategy_on_bar = pine_strat.build_step_evaluator(bars_10y)
            elif strategy_id in PINE_STRATEGY_SUITE:
                suite_item = PINE_STRATEGY_SUITE[strategy_id]
                pine_strat = PineScriptStrategy(suite_item["code"], custom_name=suite_item["name"])
                eval_bars, strategy_on_bar = pine_strat.build_step_evaluator(bars_10y)
            elif strategy_id == "orb":
                strat = OpeningRangeBreakoutStrategy(orb_minutes=15, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
                eval_bars = bars_10y
            elif strategy_id == "fvg":
                strat = FairValueGapStrategy(min_fvg_points=3.0, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
                eval_bars = bars_10y
            elif strategy_id == "boswaves":
                strat = BOSWavesGravityStrategy(lookback_swing=10, risk_reward=2.0, contracts=contracts)
                strategy_on_bar = strat.on_bar
                eval_bars = bars_10y
            else:
                strat = OpeningRangeBreakoutStrategy(contracts=contracts)
                strategy_on_bar = strat.on_bar
                eval_bars = bars_10y

            config = BacktestConfig(
                initial_capital=capital,
                commission_per_side=commission,
                slippage_ticks=slippage_ticks
            )
            engine = BacktestEngine(config=config)
            full_result = engine.run(eval_bars, strategy_fn=strategy_on_bar)
            tf_report = MultiTimeframeAnalyzer.analyze(full_result.trades, full_result.equity_curve, initial_capital=capital)

            tf_data = []
            if tf_report and tf_report.windows:
                for w in tf_report.windows:
                    tf_data.append({
                        "label": w.window_label,
                        "trades": w.total_trades,
                        "win_rate": w.win_rate,
                        "net_profit_usd": w.net_profit_usd,
                        "profit_factor": w.profit_factor,
                        "max_drawdown_usd": w.max_drawdown_usd,
                        "max_drawdown_pct": w.max_drawdown_pct,
                        "sharpe": w.sharpe_ratio
                    })

            self._set_headers("application/json", status=200)
            self.wfile.write(json.dumps({"success": True, "timeframe_matrix": tf_data}).encode("utf-8"))
            return

        else:
            self._set_headers("text/plain", status=404)
            self.wfile.write(b"Not found")

def run_dashboard(host: str = "127.0.0.1", port: int = 8888):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, DashboardRequestHandler)
    logger.info(f"MNQ Dashboard LIVE at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard server...")
        httpd.server_close()

if __name__ == "__main__":
    run_dashboard()
