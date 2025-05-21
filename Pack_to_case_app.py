import streamlit as st
import math
import os
import re
from itertools import permutations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

def pack_to_case_app():
    try:

        # Uncommented this because in local machine we need to get the output 
        st.set_page_config(page_title="Pack to Case Converter", page_icon=":page_with_curl:", layout="wide")

        main_col1, main_col2, main_col3 = st.columns([1, 1, 1])

        with main_col2:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # st.image("tcp_logo.png", use_container_width=False)
                st.markdown("<h1 style='text-align: center;'>Pack-Case Converter</h1>", unsafe_allow_html=True)

            def get_factors(n):
                """Find all triplets of factors (x, y, z) such that x * y * z == n"""
                factors = []
                for i in range(1, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        for j in range(1, int(math.sqrt(n // i)) + 1):
                            if (n // i) % j == 0:
                                k = (n // i) // j
                                factors.append((i, j, k))
                return factors

            def get_unique_orientations_with_tolerance(pack_length, pack_width, pack_height, total_packs,
                                                    length_tol=0.0, width_tol=0.0, height_tol=0.0):
                factor_combinations = get_factors(total_packs)

                result = {
                    "Length Fixed": [],
                    "Width Fixed": [],
                    "Height Fixed": []
                }

                seen_length_fixed = set()
                seen_width_fixed = set()
                seen_height_fixed = set()

                for x, y, z in factor_combinations:
                    for perm in set(permutations((x, y, z))):
                        lx, wx, hx = perm
        # Fixed Length
                        if lx * wx * hx == total_packs:
                            key = (wx, hx)
                            if key not in seen_length_fixed:
                                seen_length_fixed.add(key)
                                case_dims = (
                                    round(lx * pack_length + length_tol, 2),
                                    round(wx * pack_width + width_tol, 2),
                                    round(hx * pack_height + height_tol, 2)
                                )
                                result["Length Fixed"].append({
                                    "Orientation": (lx, wx, hx),
                                    "Case Dimensions": case_dims
                                })

                        # Fixed Width
                        if lx * 1 * hx == total_packs:
                            key = (lx, hx)
                            if key not in seen_width_fixed:
                                seen_width_fixed.add(key)
                                case_dims = (
                                    round(lx * pack_length + length_tol, 2),
                                    round(pack_width + width_tol, 2),
                                    round(hx * pack_height + height_tol, 2)
                                )
                                result["Width Fixed"].append({
                                    "Orientation": (lx, 1, hx),
                                    "Case Dimensions": case_dims
                                })

                        # Fixed Height
                        if lx * wx * 1 == total_packs:
                            key = (lx, wx)
                            if key not in seen_height_fixed:
                                seen_height_fixed.add(key)
                                case_dims = (
                                    round(lx * pack_length + length_tol, 2),
                                    round(wx * pack_width + width_tol, 2),
                                    round(pack_height + height_tol, 2)
                                )
                                result["Height Fixed"].append({
                                    "Orientation": (lx, wx, 1),
                                    "Case Dimensions": case_dims
                                })
                return result
            st.write(" ")
            st.write("*Note: All the dimensions are in inches*")
            st.markdown("<h5 style='text-align: center;'>Pack Dimensions</h5>", unsafe_allow_html=True)

            pack_col1, pack_col2, pack_col3 = st.columns(3)

            with pack_col1:
                case_length = st.number_input("Pack Length (inches)", min_value=0.1, format="%.2f")
            with pack_col2:
                case_width = st.number_input("Pack Width (inches)", min_value=0.1, format="%.2f")
                num_cases = st.number_input("Number of Packs per Case", min_value=1, step=1)
            with pack_col3:
                case_height = st.number_input("Pack Height (inches)", min_value=0.1, format="%.2f")

            # Tolerance section
            st.write(" ")
            st.markdown("<h5 style='text-align: center;'>Tolerance</h5>", unsafe_allow_html=True)

            tol_col1, tol_col2, tol_col3 = st.columns(3)
            with tol_col1:
                tol_length = st.number_input("Length Tolerance (inches)", min_value=0.0, value=0.7, format="%.2f")
            with tol_col2:
                tol_width = st.number_input("Width Tolerance (inches)", min_value=0.0, value=0.5, format="%.2f")
            with tol_col3:
                tol_height = st.number_input("Height Tolerance (inches)", min_value=0.0, value=0.5, format="%.2f")
            # if st.button("Calculate Pallet Dimensions"):
            #     pallet1, pallet2, pallet3 = calculate_pallet_dimensions(case_length, case_width, case_height, num_cases)

            #     st.write("### Case Dimensions:")
            #     st.write(f"1) Keeping Length Constant: {pallet1[0]:.2f} x {pallet1[1]:.2f} x {pallet1[2]:.2f}")
            #     st.write(f"2) Keeping Width Constant: {pallet2[0]:.2f} x {pallet2[1]:.2f} x {pallet2[2]:.2f}")
            #     st.write(f"3) Keeping Height Constant: {pallet3[0]:.2f} x {pallet3[1]:.2f} x {pallet3[2]:.2f}")
            def plot_packs_in_case(pack_dims, case_dims, orientation, title="Pack Arrangement in Case"):
                # Parse the orientation string (e.g., '1*3*5')
                if isinstance(orientation, str):
                    lx, wx, hx = map(int, orientation.split("*"))
                else:
                    lx, wx, hx = map(int, orientation)

                pack_len, pack_wid, pack_hei = pack_dims
                case_len, case_wid, case_hei = case_dims

                # Create figure and axis for 3D plotting
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')

                # Draw the outer case (brown color with no edges)
                draw_box(ax, (0, 0, 0), (case_len, case_wid, case_hei), color=(0.6, 0.3, 0.1), alpha=0.15, edge=False)

                # Draw each pack inside the case (with edges for visibility)
                for i in range(lx):
                    for j in range(wx):
                        for k in range(hx):
                            x = i * pack_len
                            y = j * pack_wid
                            z = k * pack_hei
                            draw_box(ax, (x, y, z), (pack_len, pack_wid, pack_hei), color=(0.2, 0.5, 0.8), alpha=0.8,edge=True)
            # Set axes limits
                ax.set_xlim(0, case_len)
                ax.set_ylim(0, case_wid)
                ax.set_zlim(0, case_hei)

                # Set labels and title
                ax.set_xlabel("Length (in)")
                ax.set_ylabel("Width (in)")
                ax.set_zlabel("Height (in)")
                ax.set_title(title)

                # Streamlit handles the rendering itself
                st.pyplot(fig)

            def draw_box(ax, origin, size, color=(0.2, 0.5, 0.8), alpha=0.6, edge=True):
                # Drawing function for 3D box
                x, y, z = origin
                dx, dy, dz = size

                vertices = np.array([
                    [x, y, z],
                    [x + dx, y, z],
                    [x + dx, y + dy, z],
                    [x, y + dy, z],
                    [x, y, z + dz],
                    [x + dx, y, z + dz],
                    [x + dx, y + dy, z + dz],
                    [x, y + dy, z + dz]
                ])

                faces = [
                    [vertices[i] for i in [0, 1, 2, 3]],  # Bottom
                    [vertices[i] for i in [4, 5, 6, 7]],  # Top
                    [vertices[i] for i in [0, 1, 5, 4]],  # Front
                    [vertices[i] for i in [2, 3, 7, 6]],  # Back
                    [vertices[i] for i in [1, 2, 6, 5]],  # Right
                    [vertices[i] for i in [0, 3, 7, 4]]   # Left
                ]

                # No edges for the transparent outer box (case)
                box = Poly3DCollection(faces, alpha=alpha, facecolors=color, edgecolors='black' if edge else None)
                ax.add_collection3d(box)
            # Visualize Cases
            # if st.button("Calculate & Visualize Pallet Dimensions"):

            # =========================================changed=====================================================
            result_orientations = get_unique_orientations_with_tolerance(case_length, case_width, case_height, num_cases,tol_length,tol_width,tol_height)

            # st.write(result_orientations)

            # Convert the orientation results into separate DataFrames
            df_length_fixed = pd.DataFrame(result_orientations["Length Fixed"])
            df_width_fixed = pd.DataFrame(result_orientations["Width Fixed"])
            df_height_fixed = pd.DataFrame(result_orientations["Height Fixed"])

            # Cubeness Function
            def cubeness_score(length, width, height, weight=None, max_weight=1200,
                            consider_volume=False, pallet_dims=(40, 48), max_height=60):

                dims = [length, width, height]
                max_dim = max(dims)
                min_dim = min(dims)
            # Ratio-based cubeness: closer to 1 is better
                base_cubeness = min_dim / max_dim

                # Penalize if one side is too long compared to the others
                aspect_ratios = [
                    length / width, width / length,
                    length / height, height / length,
                    width / height, height / width,
                ]
                max_aspect_ratio = max(aspect_ratios)
                if max_aspect_ratio > 2:
                    base_cubeness *= 0.7  # penalize if shape is too stretched

                # Add penalty for height over pallet limit
                if height > max_height:
                    base_cubeness *= 0.6

                # Penalize overweight cases
                if weight is not None and weight > max_weight:
                    base_cubeness *= 0.5

                # Reward efficient use of space by volume
                if consider_volume:
                    volume = length * width * height
                    base_cubeness = (volume / (max(pallet_dims[0], pallet_dims[1])*2 * max_height))

                # Ensure result stays between 0 and 1
                return round(min(max(base_cubeness, 0), 1), 4)
# Add dimension ratios
            df_length_fixed["fixed_length/width"] = df_length_fixed["Case Dimensions"].apply(lambda x: round(x[0] / x[1], 2))
            df_length_fixed["fixed_length/height"] = df_length_fixed["Case Dimensions"].apply(lambda x: round(x[0] / x[2], 2))

            df_length_fixed["Ratio of Ratios"] = df_length_fixed["Case Dimensions"].apply(
                lambda x: round((x[0] / x[1]) / (x[0] / x[2]), 2))
            df_length_fixed["Volume of Case (in^3)"] = df_length_fixed["Case Dimensions"].apply(
                lambda x: round(x[0] * x[1] * x[2], 2))
            df_length_fixed["Cubeness Score"] = df_length_fixed["Case Dimensions"].apply(
                lambda x: cubeness_score(length=float(x[0]), width=float(x[1]), height=float(x[2])))

            df_width_fixed["fixed_width/length"] = df_width_fixed["Case Dimensions"].apply(lambda x: round(x[1] / x[0], 2))
            df_width_fixed["fixed_width/height"] = df_width_fixed["Case Dimensions"].apply(lambda x: round(x[1] / x[2], 2))

            df_width_fixed["Ratio of Ratios"] = df_width_fixed["Case Dimensions"].apply(
                lambda x: round((x[1] / x[0]) / (x[1] / x[2]), 2))
            df_width_fixed["Volume of Case (in^3)"] = df_width_fixed["Case Dimensions"].apply(
                lambda x: round(x[0] * x[1] * x[2], 2))
            df_width_fixed["Cubeness Score"] = df_width_fixed["Case Dimensions"].apply(
                lambda x: cubeness_score(length=float(x[0]), width=float(x[1]), height=float(x[2])))

            df_height_fixed["fixed_height/length"] = df_height_fixed["Case Dimensions"].apply(lambda x: round(x[2] / x[0], 2))
            df_height_fixed["fixed_height/width"] = df_height_fixed["Case Dimensions"].apply(lambda x: round(x[2] / x[1], 2))

            df_height_fixed["Ratio of Ratios"] = df_height_fixed["Case Dimensions"].apply(
                lambda x: round((x[2] / x[0]) / (x[2] / x[1]), 2))
            df_height_fixed["Volume of Case (in^3)"] = df_height_fixed["Case Dimensions"].apply(
                lambda x: round(x[0] * x[1] * x[2], 2))
            df_height_fixed["Cubeness Score"] = df_height_fixed["Case Dimensions"].apply(
                lambda x: cubeness_score(length=float(x[0]), width=float(x[1]), height=float(x[2])))
            # Resetting the Index
            df_length_fixed.index = df_length_fixed.index + 1
            df_width_fixed.index = df_width_fixed.index + 1
            df_height_fixed.index = df_height_fixed.index + 1

            # Create orientation list for selectbox
            length_orientation_selectbox = []
            width_orientation_selectbox = []
            height_orientation_selectbox = []

            for u in range(len(df_length_fixed["Case Dimensions"])):
                temp_orient = "Orientation " + str(u + 1)
                length_orientation_selectbox.append(temp_orient)

            for u in range(len(df_width_fixed["Case Dimensions"])):
                temp_orient = "Orientation " + str(u + 1)
                width_orientation_selectbox.append(temp_orient)

            for u in range(len(df_height_fixed["Case Dimensions"])):
                temp_orient = "Orientation " + str(u + 1)
                height_orientation_selectbox.append(temp_orient)

            df_all = pd.concat([
                df_length_fixed.assign(Source = 'Length'),
                df_width_fixed.assign(Source = 'Width'),
                df_height_fixed.assign(Source = 'Heigth')
            ], ignore_index=True)

            # top3_scores = df_all['Cubeness Score'].nlargest(3).unique()
            # ========================This one is added=====================================================
            df_all_sorted = df_all.sort_values(by='Cubeness Score', ascending=False).reset_index(drop=True)
            top3_scores = df_all_sorted.head(3)[["Orientation", "Case Dimensions"]].values.tolist()
            # ==============================================================================================

            def highlight_global_top3(df):
                def style_row(row):
                    
                     # ========================This one is changed=====================================================
                    if  [row["Orientation"], row["Case Dimensions"]] in top3_scores:
                        return ['background-color: lightgreen'] * len(row)
                    else:
                        return [''] *len(row)
                return df.style.apply(style_row,axis=1)

            # Visualization
            st.write("### Pack configurations inside the Case:")

            df_length_fixed = df_length_fixed.sort_values(by='Cubeness Score', ascending=False).reset_index(drop=True)

            with main_col1:
                for i in range(45):
                    st.write(" ")

                st.write("1) Keeping Length Constant (LxWxH):") 

                 # =======================================changed=====================================================
                st.dataframe(highlight_global_top3(df_length_fixed[["Orientation", "Case Dimensions"]]), use_container_width=True)
                # st.write(type(df1))

                selected_length_orientation = st.selectbox("Select an Orientation to Visualize", length_orientation_selectbox, key="length")
                selected_length_number = int(selected_length_orientation.split()[-1]) - 1

                # =========================================changed=====================================================
                plot_packs_in_case([case_length, case_width, case_height],
                                df_length_fixed["Case Dimensions"].iloc[selected_length_number],
                                df_length_fixed["Orientation"].iloc[selected_length_number],
                                "Keeping Length Constant")
            
            df_width_fixed = df_width_fixed.sort_values(by='Cubeness Score', ascending=False).reset_index(drop=True)

            with main_col2:
                st.write("2) Keeping Width Constant (LxWxH):")

                 # =========================================changed=====================================================
                st.dataframe(highlight_global_top3(df_width_fixed[["Orientation","Case Dimensions"]]), use_container_width=True)

                selected_width_orientation = st.selectbox("Select an Orientation to Visualize", width_orientation_selectbox, key="width")
                selected_width_number = int(selected_width_orientation.split()[-1]) - 1

                plot_packs_in_case([case_length, case_width, case_height],
                                df_width_fixed["Case Dimensions"].iloc[selected_width_number],
                                df_width_fixed["Orientation"].iloc[selected_width_number],
                                "Keeping Width Constant")
            
            df_height_fixed = df_height_fixed.sort_values(by='Cubeness Score', ascending=False).reset_index(drop=True)

            with main_col3:
                for i in range(45):
                    st.write(" ")

                st.write("3) Keeping Height Constant (LxWxH):")
                # =========================================changed=====================================================
                st.dataframe(highlight_global_top3(df_height_fixed[["Orientation","Case Dimensions"]]), use_container_width=True)

                selected_height_orientation = st.selectbox("Select an Orientation to Visualize", height_orientation_selectbox, key="height")
                selected_height_number = int(selected_height_orientation.split()[-1]) - 1

                plot_packs_in_case([case_length, case_width, case_height],
                                df_height_fixed["Case Dimensions"].iloc[selected_height_number],
                                df_height_fixed["Orientation"].iloc[selected_height_number],
                                "Keeping Height Constant")
    except Exception as e:
        st.error(e)

pack_to_case_app()