from financial_assistant.tools.market_data import get_market_data


def main():
    for symbol in ["XAU/USD", "EUR/USD"]:
        print(f"\nFetching {symbol}...")

        df = get_market_data(
            symbol=symbol,
            interval="1h",
            outputsize=100,
        )

        print(df.tail())
        print(f"\nRows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    main()
