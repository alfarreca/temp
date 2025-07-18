from pathlib import Path
import re

# Load the previously updated script with price trend % already added
updated_path = "/mnt/data/Price_Tracker22_FIXED_pct.py"
with open(updated_path, "r") as file:
    script_with_price_pct = file.read()

# New code block for Normalized Performance tab with enhanced legend
new_tab1_code = '''
                with tabs[1]:
                    st.subheader("📊 Normalized Performance (Start = 100)")
                    norm_chart = go.Figure()
                    start_values = norm_df.iloc[:, 0]
                    last_values = norm_df.iloc[:, -1]

                    total_pct_change = ((last_values - start_values) / start_values) * 100
                    normed_pct_change = norm_df.divide(start_values, axis=0) * 100
                    pct_change_from_start = norm_df.subtract(start_values, axis=0).divide(start_values, axis=0) * 100

                    for sym in normed.index:
                        change = total_pct_change[sym]
                        label_name = f"{sym} ({change:+.2f}%)"
                        norm_chart.add_trace(go.Scatter(
                            x=labels,
                            y=(normed_pct_change.loc[sym]),
                            customdata=pct_change_from_start.loc[sym].values.reshape(-1, 1),
                            mode="lines",
                            name=label_name,
                            hovertemplate=(
                                f"<b>{sym}</b><br>"
                                + "Normalized: %{y:.2f}<br>"
                                + "Change: %{customdata[0]:.2f}%"
                            )
                        ))
                    norm_chart.update_layout(hovermode="x unified", height=500)
                    st.plotly_chart(norm_chart, use_container_width=True)
'''

# Replace old tab[1] block with the new one
final_script = re.sub(
    r"with tabs\[1\]:.*?st\.plotly_chart\(norm_chart, use_container_width=True\)",
    new_tab1_code.strip(),
    script_with_price_pct,
    flags=re.DOTALL
)

# Save the final merged script
final_script_path = "/mnt/data/Price_Tracker22_FINAL.py"
Path(final_script_path).write_text(final_script)
final_script_path
