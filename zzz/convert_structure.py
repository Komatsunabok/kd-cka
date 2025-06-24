def txt_to_markdown_tables(txt_lines):
    tables = []
    current_model = None
    current_lines = []
    for line in txt_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # コロンが含まれていない行をモデル名とみなす
        if ':' not in stripped:
            if current_model and current_lines:
                tables.append((current_model, current_lines))
            current_model = stripped
            current_lines = []
        else:
            current_lines.append(stripped)
    # 最後のモデルも追加
    if current_model and current_lines:
        tables.append((current_model, current_lines))
    return tables

def lines_to_table(lines):
    table = "| 層番号 | 層タイプ           | 出力サイズ                |\n"
    table += "|--------|-------------------|---------------------------|\n"
    for line in lines:
        idx_type, size = line.split(':', 1)
        idx, layer_type = idx_type.split('_', 1)
        size = size.strip()
        table += f"| {idx}      | {layer_type:<17} | {size} |\n"
    return table

if __name__ == "__main__":
    input_file = "model_structure.txt"
    output_file = "model_structure.md"

    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    tables = txt_to_markdown_tables(lines)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# モデル構造一覧\n\n")
        for model, layer_lines in tables:
            print(f"Processing model: {model}")
            f.write(f"## {model}\n\n")
            f.write(lines_to_table(layer_lines))
            f.write("\n")

    print("変換が完了しました。model_structure.md をご確認ください。")