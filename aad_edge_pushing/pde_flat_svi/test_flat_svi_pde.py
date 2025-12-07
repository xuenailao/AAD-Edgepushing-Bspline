import numpy as np
from aad_edge_pushing.pde_flat_svi.pde_flat_svi_aad_edgepushing import FlatSVI_PDE_AAD

def test_flat_svi_pde():
    # === Step 1. 初始化模型 ===
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.02

    # 定义 3 个 flat SVI slices（总方差）
    T_slices = [0.25, 0.5, 1.0]
    w_slices = [0.04, 0.05, 0.10]

    model = FlatSVI_PDE_AAD(S0, K, T, r, T_slices, w_slices, M=45, N_base=30)

    # === Step 2. 运行 PDE solver ===
    result = model.solve_pde_with_aad(S0_val=S0, compute_hessian=True, verbose=True)

    # === Step 3. 打印结果 ===
    print("==== Flat SVI PDE Result ====")
    # Use .get to avoid KeyError
    price = result.get('price', None)
    delta = result.get('delta', None)
    gamma = result.get('gamma', None)
    # Use runtime_sec instead of time_ms
    runtime = result.get("runtime_sec", "NA")
    if isinstance(runtime, (float, int)):
        runtime_str = f"{runtime:.2f} s"
    else:
        runtime_str = runtime
    hessian = result.get('hessian', None)
    hessian_labels = result.get('hessian_labels', None)
    w_sensitivities = result.get('w_sensitivities', None)

    if price is not None:
        print(f"Option price: {price:.6f}")
    if delta is not None:
        print(f"Delta:        {delta:.6f}")
    if gamma is not None:
        print(f"Gamma:        {gamma:.6f}")
    if hessian is not None:
        print("Hessian matrix (rows/cols):", hessian_labels if hessian_labels is not None else [])
        print(hessian)
    if runtime_str is not None:
        print(f"Runtime:      {runtime_str}")
    if w_sensitivities is not None:
        print("w_sensitivities:")
        for i, sens in enumerate(w_sensitivities):
            print(f"  Slice {i}: {sens:.6f}")

    # === Step 4. 自动生成 Markdown 报告 ===
    import pandas as pd
    import os
    md_lines = []
    md_lines.append("# Flat SVI PDE Result\n")
    md_lines.append("## Option Price and Sensitivities\n")
    # Table for price, delta, gamma, runtime
    table_rows = []
    table_rows.append(["Option Price", f"{price:.6f}" if price is not None else "N/A"])
    table_rows.append(["Delta", f"{delta:.6f}" if delta is not None else "N/A"])
    table_rows.append(["Gamma", f"{gamma:.6f}" if gamma is not None else "N/A"])
    table_rows.append(["Runtime", runtime_str if runtime_str is not None else "N/A"])
    table_rows.append(["Grid (M, N)", f"{model.M}, {model.N_base}"])
    df_main = pd.DataFrame(table_rows, columns=["Metric", "Value"])
    md_lines.append(df_main.to_markdown(index=False))
    md_lines.append("")

    # Hessian matrix (if exists)
    if hessian is not None and hessian_labels is not None:
        md_lines.append("## Hessian Matrix\n")
        # Try to pretty print using pandas
        try:
            df_hess = pd.DataFrame(hessian, columns=hessian_labels, index=hessian_labels)
            md_lines.append(df_hess.to_markdown())
        except Exception:
            md_lines.append("Hessian matrix could not be formatted as table.")
    elif hessian is not None:
        md_lines.append("## Hessian Matrix\n")
        md_lines.append("Hessian matrix present but labels are missing.\n")
        try:
            df_hess = pd.DataFrame(hessian)
            md_lines.append(df_hess.to_markdown())
        except Exception:
            md_lines.append("Hessian matrix could not be formatted as table.")

    # w_sensitivities
    if w_sensitivities is not None:
        md_lines.append("## Slice Sensitivities\n")
        w_rows = []
        for i, sens in enumerate(w_sensitivities):
            w_rows.append([f"Slice {i}", f"{sens:.6f}"])
        df_w = pd.DataFrame(w_rows, columns=["Slice", "Sensitivity"])
        md_lines.append(df_w.to_markdown(index=False))
        md_lines.append("")

    # Write to markdown file in the same directory as this script
    md_report_path = os.path.join(os.path.dirname(__file__), "flat_svi_hessian_report.md")
    with open(md_report_path, "w") as f_md:
        f_md.write("\n".join(md_lines))
    print(f"Markdown report generated: flat_svi_hessian_report.md")


if __name__ == "__main__":
    test_flat_svi_pde()