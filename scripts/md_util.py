"""Bang markdown khong can tabulate."""


def md_table(df, fmt="{:.3g}", index=False):
    if index:
        df = df.reset_index()
    cell = lambda v: fmt.format(v) if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)
