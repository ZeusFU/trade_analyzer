import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# TraderProfiler class (same as before, but without tkinter)
class TraderProfiler:
    def __init__(self, trades, initial_balance, daily_loss_limit, has_previous_account=False):
        self.trades = trades
        self.initial_balance = float(initial_balance)
        self.daily_loss_limit = float(daily_loss_limit)
        self.has_previous_account = has_previous_account
        self.style_categories = {
            'Scalper': {'max_duration': 10, 'min_trades': 8, 'max_points': 15},
            'Day Trader': {'max_duration': 60, 'min_trades': 3, 'max_points': 50},
            'Swing Trader': {'max_duration': 1440, 'min_trades': 1, 'max_points': 200}
        }
        self._preprocess_data()
        self.account_age_days = self._calculate_account_age()
        self.survival_bonus = self._calculate_survival_bonus()

    def _preprocess_data(self):
        currency_cols = ['Avg. Entry', 'Avg. Close', 'Net Profit', 'Gross Profit']
        for col in currency_cols:
            self.trades[col] = self.trades[col].replace('[^0-9.-]', '', regex=True).astype(float)
        
        self.trades['Duration (min)'] = (self.trades['Close Time'] - 
                                       self.trades['Open Time']).dt.total_seconds() / 60
        self.trades['Day'] = self.trades['Open Time'].dt.normalize()
        self._validate_market_hours()

    def _validate_market_hours(self):
        same_day = (self.trades['Open Time'].dt.date == 
                   self.trades['Close Time'].dt.date)
        if not same_day.all():
            invalid_trades = self.trades[~same_day]
            raise ValueError(f"{len(invalid_trades)} trades cross market close")
    
    def _calculate_account_age(self):
        if self.trades.empty:
            return 0
        active_days = (self.trades['Day'].max() - self.trades['Day'].min()).days + 1
        return min(active_days, 30)

    def _calculate_survival_bonus(self):
        if self.account_age_days >= 21:
            return min((self.account_age_days - 14) * 2, 50)
        return 0

    def analyze(self):
        analysis = {
            'style': self._determine_trading_style(),
            'risk_profile': self._calculate_risk_profile(),
            'consistency': self._determine_consistency(),
            'efficiency': self._calculate_strategy_efficiency(),
            'survival': self._calculate_survival_metrics(),
            'intraday': self._calculate_intraday_performance(),
            'strategy_health': self._assess_strategy_health()
        }
        analysis['composite_profile'] = self._create_composite_profile(analysis)
        return analysis

    def _determine_trading_style(self):
        metrics = {
            'avg_duration': self.trades['Duration (min)'].mean(),
            'avg_points': self.trades['Points'].mean(),
            'daily_trades': self.trades.groupby('Day').size().mean()
        }
        
        for style, params in self.style_categories.items():
            if (metrics['avg_duration'] <= params['max_duration'] and
                metrics['daily_trades'] >= params['min_trades'] and
                metrics['avg_points'] <= params['max_points']):
                return {
                    'style': style,
                    'metrics': metrics,
                    'confidence': self._style_confidence(metrics, params)
                }
        return {'style': 'Hybrid/Unclear', 'metrics': metrics, 'confidence': 0}

    def _style_confidence(self, metrics, params):
        duration_conf = 1 - (metrics['avg_duration'] / params['max_duration'])
        trades_conf = metrics['daily_trades'] / params['min_trades']
        points_conf = 1 - (metrics['avg_points'] / params['max_points'])
        return min((duration_conf + trades_conf + points_conf) / 3 * 100, 100)

    def _calculate_risk_profile(self):
        daily_pnl = self.trades.groupby('Day')['Net Profit'].sum()
        loss_breaches = self._calculate_dll_breaches()  # Use inverse buffer
        max_drawdown = self._calculate_max_drawdown()
        recent_risk = self._calculate_recent_risk(daily_pnl)
        rr_ratio = self._risk_reward_ratio()
        
        # Calculate DLL bonus (reward for staying far below DLL)
        dll_bonus = self._calculate_dll_bonus()
        
        # Calculate base risk score
        risk_score = min(
            (loss_breaches * 25) + 
            (max_drawdown * 0.5) + 
            (recent_risk * 1.5) + 
            (rr_ratio * 10), 
            100
        )
        
        # Add DLL bonus
        risk_score += dll_bonus
        risk_score = min(risk_score, 100)  # Cap at 100
        
        return {
            'category': self._risk_category(risk_score),
            'score': risk_score,
            'metrics': {
                'loss_breaches': loss_breaches,
                'max_drawdown': max_drawdown,
                'recent_risk': recent_risk,
                'rr_ratio': rr_ratio,
                'dll_bonus': dll_bonus  # Show bonus points
            }
        }

    def _calculate_dll_breaches(self, buffer_pct=0.1):
        # Calculate inverse buffer threshold
        breach_threshold = -self.daily_loss_limit * (1 - buffer_pct)
        
        # Group trades by day and calculate daily P&L
        daily_pnl = self.trades.groupby('Day')['Net Profit'].sum()
        
        # Count breaches (daily loss >= breach_threshold)
        breaches = daily_pnl.le(breach_threshold).sum()
        return breaches

    def _calculate_dll_bonus(self, buffer_pct=0.1):
        # Calculate inverse buffer threshold
        breach_threshold = -self.daily_loss_limit * (1 - buffer_pct)
        
        # Group trades by day and calculate daily P&L
        daily_pnl = self.trades.groupby('Day')['Net Profit'].sum()
        
        # Calculate how far below the DLL the trader stayed
        distance_below_dll = ((-daily_pnl) / self.daily_loss_limit).clip(0, 1)
        
        # Reward points based on distance below DLL
        bonus = distance_below_dll.mean() * 10  # Max bonus of 10 points
        return bonus

    def _calculate_max_drawdown(self):
        # Calculate running equity
        running_equity = self.initial_balance + self.trades['Net Profit'].cumsum()
        
        # Track rolling peak
        peak = running_equity.expanding(min_periods=1).max()
        
        # Calculate drawdown from peak
        drawdown = (running_equity - peak) / peak * 100
        
        # Return worst drawdown
        return drawdown.min()

    def _risk_reward_ratio(self):
        wins = self.trades[self.trades['Net Profit'] > 10]  # Ignore tiny wins
        losses = self.trades[self.trades['Net Profit'] < -10]  # Ignore tiny losses
        
        if len(losses) == 0:
            return float('inf')  # No losses
        
        avg_win = wins['Net Profit'].mean()
        avg_loss = losses['Net Profit'].abs().mean()
        return avg_win / avg_loss

    def _calculate_recent_risk(self, daily_pnl):
        if len(daily_pnl) < 5:
            return 0
        last_week = daily_pnl[-5:].mean()
        first_week = daily_pnl[:5].mean()
        return (first_week - last_week) / abs(first_week) * 100 if first_week != 0 else 0

    def _risk_category(self, score):
        if score >= 90: return 'Exceptional Risk Management'
        if score >= 75: return 'Aggressive Risk'
        if score >= 50: return 'Moderate Risk'
        if score >= 25: return 'Conservative Risk'
        return 'Risk Averse'

    def _determine_consistency(self):
        daily_pnl = self.trades.groupby('Day')['Net Profit'].sum()
        if len(daily_pnl) < 5:
            return {
                'win_rate': None,
                'volatility': None,
                'category': 'Insufficient Data (Min 5 Days)'
            }
        
        win_rate = daily_pnl.gt(0).mean() * 100
        normalized_volatility = daily_pnl.std() / self.initial_balance * 100
        
        return {
            'win_rate': win_rate,
            'volatility': normalized_volatility,
            'category': self._consistency_category(win_rate, normalized_volatility)
        }

    def _consistency_category(self, win_rate, norm_volatility):
        if pd.isnull(win_rate):
            return 'Insufficient Data'
            
        if win_rate >= 70 and norm_volatility <= 5:
            return 'Exceptionally Consistent'
        if win_rate >= 60 and norm_volatility <= 10:
            return 'Reliably Consistent'
        if win_rate >= 50 and norm_volatility <= 15:
            return 'Moderately Consistent'
        return 'Erratic Performance'

    def _calculate_strategy_efficiency(self):
        sharpe = self._calculate_sharpe()
        sortino = self._calculate_sortino()
        profit_factor = self._calculate_profit_factor()
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'profit_factor': profit_factor,
            'efficiency_class': self._efficiency_class(sharpe, sortino, profit_factor)
        }

    def _calculate_sharpe(self):
        daily_returns = self.trades.groupby('Day')['Net Profit'].sum() / self.initial_balance
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0
        return daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    def _calculate_sortino(self):
        daily_returns = self.trades.groupby('Day')['Net Profit'].sum()
        downside = daily_returns[daily_returns < 0].std()
        if downside == 0:
            return 0
        return daily_returns.mean() / downside * np.sqrt(252)

    def _calculate_profit_factor(self):
        gross_profit = self.trades[self.trades['Net Profit'] > 0]['Net Profit'].sum()
        gross_loss = abs(self.trades[self.trades['Net Profit'] < 0]['Net Profit'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _efficiency_class(self, sharpe, sortino, pf):
        if sharpe > 2 and sortino > 3 and pf > 2.5: return 'Elite Efficiency'
        if sharpe > 1.5 and sortino > 2 and pf > 1.5: return 'High Efficiency'
        if sharpe > 1 and sortino > 1.5 and pf > 1: return 'Moderate Efficiency'
        return 'Low Efficiency'

    def _calculate_survival_metrics(self):
        active_days = self.trades['Day'].nunique()
        return {
            'age_days': self.account_age_days,
            'active_days': active_days,
            'survival_bonus': self.survival_bonus,
            'consistency': self._calculate_active_day_consistency()
        }

    def _calculate_active_day_consistency(self):
        active_days = self.trades.groupby('Day').size().gt(0)
        max_streak = active_days.astype(int).groupby((~active_days).cumsum()).cumcount().max()
        return max_streak

    def _calculate_intraday_performance(self):
        self.trades['Hour'] = self.trades['Open Time'].dt.hour
        hourly = self.trades.groupby('Hour').agg({
            'Net Profit': ['mean', 'count', 'sum'],
            'Points': 'mean'
        })
        hourly.columns = ['Avg Profit', 'Trade Count', 'Total Profit', 'Avg Points']
        return hourly.reset_index().to_dict(orient='records')

    def _assess_strategy_health(self):
        daily_pnl = self.trades.groupby('Day')['Net Profit'].sum()
        if len(daily_pnl) < 5:
            return {'status': 'Insufficient Data'}
            
        rolling_5d = daily_pnl.rolling(5).mean().dropna()
        decay_score = (rolling_5d.iloc[-1] - rolling_5d.iloc[0]) / abs(rolling_5d.iloc[0]) * 100
        
        return {
            'rolling_5d_mean': rolling_5d.tolist(),
            'decay_score': decay_score,
            'status': 'Degrading' if decay_score < -20 else 'Stable'
        }

    def _create_composite_profile(self, analysis):
        style = analysis['style']['style']
        risk = analysis['risk_profile']['category']
        survival_tier = "Veteran" if self.account_age_days >= 25 else \
                       "Seasoned" if self.account_age_days >= 18 else \
                       "Developing" if self.account_age_days >= 10 else "Rookie"
        consistency = "Stable" if analysis['survival']['consistency'] >= 5 else \
                     "Volatile" if analysis['survival']['consistency'] >= 3 else "Erratic"
        return f"{style} {risk} {survival_tier} ({consistency})"


# Streamlit App
def main():
    st.title("Trader Performance Analyzer")
    st.markdown("Upload your trading journal CSV to analyze your performance.")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        trades = pd.read_csv(uploaded_file, parse_dates=['Open Time', 'Close Time'])
        
        # Input fields
        st.sidebar.header("Account Parameters")
        initial_balance = st.sidebar.number_input("Initial Balance ($)", min_value=0.0, value=10000.0)
        daily_loss_limit = st.sidebar.number_input("Daily Loss Limit ($)", min_value=0.0, value=1000.0)
        has_previous_account = st.sidebar.radio("Previously Lost an Account?", ["No", "Yes"]) == "Yes"

        # Analyze button
        if st.sidebar.button("Analyze"):
            try:
                profiler = TraderProfiler(trades, initial_balance, daily_loss_limit, has_previous_account)
                results = profiler.analyze()

                # Display results
                st.header("Analysis Results")

                # Composite Profile
                st.subheader("Composite Profile")
                st.write(results['composite_profile'])

                # Trading Style
                st.subheader("Trading Style")
                st.write(f"**Style**: {results['style']['style']} ({results['style']['confidence']:.1f}% confidence)")
                st.write(f"**Avg Duration**: {results['style']['metrics']['avg_duration']:.1f} mins")
                st.write(f"**Daily Trades**: {results['style']['metrics']['daily_trades']:.1f}")
                st.write(f"**Avg Points**: {results['style']['metrics']['avg_points']:.1f}")

                # Risk Analysis
                st.subheader("Risk Analysis")
                st.write(f"**Category**: {results['risk_profile']['category']}")
                st.write(f"**Loss Breaches**: {results['risk_profile']['metrics']['loss_breaches']}")
                st.write(f"**Max Drawdown**: {results['risk_profile']['metrics']['max_drawdown']:.1f}%")
                st.write(f"**Risk/Reward**: {results['risk_profile']['metrics']['rr_ratio']:.2f}:1")
                st.write(f"**DLL Bonus**: +{results['risk_profile']['metrics']['dll_bonus']:.1f} pts")

                # Consistency
                st.subheader("Consistency")
                st.write(f"**Win Rate**: {results['consistency']['win_rate']:.1f}%")
                st.write(f"**Volatility**: {results['consistency']['volatility']:.1f}%")
                st.write(f"**Category**: {results['consistency']['category']}")

                # Survival Metrics
                st.subheader("Survival Metrics")
                st.write(f"**Account Age**: {results['survival']['age_days']} days")
                st.write(f"**Active Days**: {results['survival']['active_days']}")
                st.write(f"**Longest Streak**: {results['survival']['consistency']} days")
                st.write(f"**Survival Bonus**: +{results['survival']['survival_bonus']} pts")

                # Strategy Health
                st.subheader("Strategy Health")
                st.write(f"**Status**: {results['strategy_health']['status']}")
                st.write(f"**5-Day Trend**: {[f'{x:.2f}' for x in results['strategy_health']['rolling_5d_mean']]}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
