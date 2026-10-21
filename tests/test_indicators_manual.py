from financial_assistant.tools.market_data import get_market_data
from financial_assistant.tools.technical_indicators import (
    add_technical_indicators,
)


def main():
    for symbol in ["XAU/USD", "EUR/USD"]:
        print(f"\nAnalyzing {symbol}...")

        market_data = get_market_data(
            symbol=symbol,
            interval="1h",
            outputsize=100,
        )

        analyzed_data = add_technical_indicators(market_data)

        columns = [
            "datetime",
            "close",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
        ]

        print(analyzed_data[columns].tail(5).to_string(index=False))

        latest = analyzed_data.iloc[-1]

        print("\nLatest indicator values:")
        print(f"Close: {latest['close']:.5f}")
        print(f"RSI (14): {latest['rsi_14']:.2f}")
        print(f"MACD: {latest['macd']:.6f}")
        print(f"MACD Signal: {latest['macd_signal']:.6f}")


if __name__ == "__main__":
    main()