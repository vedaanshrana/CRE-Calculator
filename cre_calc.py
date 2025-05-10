import streamlit as st
import numpy as np
import math
import re
from itertools import permutations
from scipy.integrate import quad
from sympy import symbols, sympify, lambdify
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


# Utility function for evaluating expressions
def parse_rate_law(rate_expr, variables):
    try:
        # Create symbols for all variables
        sym_vars = symbols(' '.join(variables))
        expr = sympify(rate_expr)

        # If there's only one variable, don't put it in a tuple
        if len(variables) == 1:
            return lambdify(sym_vars, expr, modules=['numpy'])
        else:
            return lambdify(sym_vars, expr, modules=['numpy'])
    except Exception as e:
        st.error(f"Invalid rate law expression: {str(e)}")
        return None


# Conversion Calculator Functions
def conv_cstr(k, tau):
    return (k * tau) / (1 + k * tau)


def conv_pfr(k, tau):
    return 1 - math.exp(-k * tau)


def conv_batch(k, tau):
    return 1 - math.exp(-k * tau)


def conv_pbr(k, tau):
    return 1 - math.exp(-k * tau)


# Reaction Parser Function
def parse_reaction(reaction):
    def parse_side(side_str, sign):
        species_dict = {}
        species_list = []
        terms = side_str.split('+')
        for term in terms:
            term = term.strip()
            match = re.match(r"^(\d*)\s*([A-Za-z]\w*)$", term)
            if match:
                coeff_str, species = match.groups()
                coeff = int(coeff_str) if coeff_str else 1
                species_dict[species] = sign * coeff
                species_list.append(species)
            else:
                raise ValueError(f"Invalid species format: '{term}'")
        return species_list, species_dict

    if '->' not in reaction:
        raise ValueError("Reaction must contain '->' to separate reactants and products")

    left, right = reaction.split('->')
    reactants, reactant_dict = parse_side(left, -1)
    products, product_dict = parse_side(right, 1)

    # Merge both dicts
    stoichiometry = {**reactant_dict, **product_dict}

    return reactants, products, stoichiometry


# Streamlit Config
st.set_page_config(page_title="Chemical Reactor Engineering Calculator", layout="wide")

# Custom background (neon green)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #39ff14;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Title
st.title("Chemical Reactor Engineering Calculator")
st.logo("logo.png", size = "large")

# Sidebar Menu
mode = st.sidebar.radio(
    "Select Operation",
    [
        "Welcome",
        "Volume Calculator",
        "Optimisation"
    ]
)

# Welcome Screen
if mode == "Welcome":
    st.subheader("applications in industrial and scientific use-cases")

    st.divider()
    st.write("Use the sidebar to choose an operation.")

# MODULE 1: Volume Calculator
elif mode == "Volume Calculator":
    st.subheader("Volume Calculator")
    reaction = st.text_input("Enter the chemical reaction (eg: A + B -> C + D)", value="A -> B")

    try:
        reactants, products, stoich = parse_reaction(reaction)
    except ValueError as e:
        st.error(f"Error parsing reaction: {str(e)}")
        st.stop()

    r_phase = st.selectbox("Select Phase of the Reaction", ["Liquid Phase", "Gas Phase"])
    v_0 = st.number_input("Initial Volumetric Flow Rate [v_0] (in L/s)", value=1.0)
    MolarFlowRates = []
    theta = []
    stoich_ratios = []

    for i in range(len(reactants)):
        MolarFlowRates.append(
            st.number_input(f"Enter Initial Molar Flow Rate of {reactants[i]} [F_{reactants[i]}0] (in mol/s)",
                            value=1.0, key=f"mfr_{i}"))
        theta.append(MolarFlowRates[i] / MolarFlowRates[0])
        stoich_ratios.append((-1) * stoich[reactants[i]] / stoich[reactants[0]])

    for j in range(len(products)):  # adds 0 to products mfr and updates the stoich ratios
        theta.append(0)
        stoich_ratios.append((stoich[products[j]] / stoich[reactants[0]]) * (-1))

    rtype = st.pills("Reactor Type", ["CSTR", "Batch", "PFR", "PBR"])
    rate_expr = st.text_input("Enter rate law [-r_A]", value="0.2*C_A")
    X = st.slider("Conversion (X)", 0.0, 1.0, 0.6)

    # Calculate basic parameters
    F_A0 = MolarFlowRates[0]
    C_A0 = F_A0 / v_0  # Initial concentration of A

    # Get all species and their concentration variables
    all_species = reactants + products
    conc_vars = [f"C_{s}" for s in all_species]
    rate_func = parse_rate_law(rate_expr, conc_vars)

    if not rate_func:
        st.stop()

    # Calculate delta and epsilon
    DELTA = (sum(stoich_ratios) / stoich_ratios[0])*(-1)
    EPSILON = (MolarFlowRates[0] / sum(MolarFlowRates)) * DELTA if r_phase == "Gas Phase" else 0
    print(f'delta: {DELTA}')
    print(f'epsilon: {EPSILON}')
    print(f'theta: {theta}')
    print(f'Conc: {conc_vars}')

    try:
        if rtype == "CSTR":
            # For CSTR, we need to calculate exit concentrations
            # First calculate the volumetric flow rate at exit
            v = v_0 * (1 + EPSILON * X)

            # Calculate exit molar flow rates
            F_exit = {}
            for i, species in enumerate(all_species):
                F_exit[species] = F_A0 * (theta[i] + stoich_ratios[i] * X)

            # Calculate exit concentrations
            exit_concentrations = {}
            for species in all_species:
                exit_concentrations[f"C_{species}"] = F_exit[species] / v

            # Prepare arguments for rate function
            args = [exit_concentrations.get(var, 0) for var in conc_vars]
            r_A = rate_func(*args)

            if r_A == 0:
                st.error("Rate function returns zero. Division by zero.")
            else:
                V = (F_A0 * X) / r_A
                st.success(f"Reactor Volume V = {V:.3f} L")
            print(f'rate cstr: {rate_func(*args)}')

        elif rtype in ["PFR", "PBR"]:
            def integrand(X):
                # Calculate all concentrations at this conversion
                current_C = []
                for k in range(len(theta)):
                    current_C.append((F_A0 * (theta[k] + stoich_ratios[k] * X)) / (v_0 * (1 + EPSILON * X)))

                # Prepare concentration values
                conc_values = {f"C_{species}": current_C[i] for i, species in enumerate(all_species)}
                args = [conc_values.get(var, 0) for var in conc_vars]
                val = rate_func(*args)
                if val == 0:
                    return float('inf')
                return 1 / val


            V_F, _ = quad(integrand, 0, X)
            V = F_A0 * V_F
            st.success(f"Reactor Volume V = {V:.3f} L")


        elif rtype == "Batch":
            # For batch reactor, we need to relate all concentrations to C_A
            # This assumes constant volume batch reactor
            def integrand(C_A):
                # Calculate all other concentrations based on C_A
                conc_values = {'C_A': C_A}
                X_batch = 1 - C_A / C_A0

                for i, species in enumerate(all_species[1:], 1):
                    C_i = (F_A0 * (theta[i] - stoich_ratios[i] * X_batch)) / (v_0 * (1 + EPSILON * X_batch))
                    conc_values[f"C_{species}"] = C_i

                args = [conc_values.get(var, 0) for var in conc_vars]
                val = rate_func(*args)
                if val == 0:
                    return float('inf')
                return 1 / val


            C_final = C_A0 * (1 - X)
            t, _ = quad(integrand, C_final, C_A0)
            st.success(f"Reaction Time = {t:.3f} s")

    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")

#MODULE 3: Optimisation
elif mode == "Optimisation":
    st.subheader("Reactor Sequence Optimisation")
    st.caption("Upload your .csv file of X vs r_A (values should be negative)")

    # Step 1: Take input of F_A0 from user
    F_A0 = st.number_input("Enter initial molar flow rate F_A0 (mol/s)", value=1.0, min_value=0.0)

    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        try:
            # Read the CSV file
            dataframe = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(dataframe)

            # Check required columns exist
            if 'X' not in dataframe.columns or 'r_A' not in dataframe.columns:
                st.error("CSV file must contain 'X' and 'r_A' columns")
                st.stop()

            # Convert r_A to negative values if not already
            if (dataframe['r_A'] > 0).any():
                st.warning("Assuming r_A values represent -r_A (converting to negative)")
                dataframe['r_A'] = -abs(dataframe['r_A'])

            # Step 2: Plot the Levenspiel plot (F_A0/(-r_A) vs X)
            dataframe['F_A0/-r_A'] = F_A0 / (-dataframe['r_A'])  # Using negative of r_A

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dataframe['X'], dataframe['F_A0/-r_A'], 'b-', linewidth=2)
            ax.set_xlabel('Conversion (X)')
            ax.set_ylabel('F_A0 / -r_A (L)')
            ax.set_title('Levenspiel Plot')
            ax.grid(True)
            st.pyplot(fig)

            # Step 3: Take input for number of reactors
            num_reactors = st.number_input("Number of reactors in sequence",
                                           min_value=1, max_value=5, value=1)

            reactors = []
            total_volume = 0

            # Step 4: Get reactor details
            st.subheader("Enter Reactor Details")
            for i in range(num_reactors):
                col1, col2 = st.columns(2)
                with col1:
                    reactor_type = st.selectbox(
                        f"Reactor {i + 1} Type",
                        ["CSTR", "PFR/PBR"],
                        key=f"type_{i}"
                    )
                with col2:
                    volume = st.number_input(
                        f"Volume of Reactor {i + 1} (L)",
                        min_value=0.0,
                        value=24.0,  # Default to test case value
                        key=f"vol_{i}"
                    )
                reactors.append({"type": reactor_type, "volume": volume})
                total_volume += volume

            # Step 5 & 6: Calculate maximum conversion
            if st.button("Calculate Maximum Conversion"):
                current_X = 0.0
                cumulative_area = 0.0  # Initialize cumulative_area

                for i, reactor in enumerate(reactors):
                    if reactor['type'] == "CSTR":
                        # Create a fine grid of possible X values between current_X and max X
                        max_X = dataframe['X'].max()
                        possible_X = np.linspace(current_X, max_X, 1000)

                        # Interpolate -r_A values for these X values
                        interp_rA = np.interp(possible_X, dataframe['X'], dataframe['r_A'])

                        # Calculate required volumes for all possible X
                        required_volumes = F_A0 * (possible_X - current_X) / (-interp_rA)

                        # Find all X where volume <= reactor volume
                        valid_indices = np.where(required_volumes <= reactor['volume'])[0]

                        if len(valid_indices) == 0:
                            st.warning(f"Cannot achieve full volume for CSTR {i + 1} with current data")
                            break

                        # Find the maximum X in valid indices
                        max_index = valid_indices[-1]
                        new_X = possible_X[max_index]
                        used_volume = required_volumes[max_index]

                        # If we're at the end of our data range but could use more volume
                        if max_index == len(possible_X) - 1 and used_volume < reactor['volume']:
                            st.warning(
                                f"CSTR {i + 1} could potentially achieve higher conversion with more data points")

                        cumulative_area += used_volume
                        current_X = new_X

                    elif reactor['type'] == "PFR/PBR":
                        # Calculate area under curve for PFR
                        subset = dataframe[(dataframe['X'] >= current_X)]
                        if len(subset) < 2:
                            break

                        area_used = 0
                        prev_X = current_X
                        prev_val = F_A0 / (-subset.iloc[0]['r_A'])

                        for idx in range(1, len(subset)):
                            curr_X = subset.iloc[idx]['X']
                            curr_val = F_A0 / (-subset.iloc[idx]['r_A'])

                            # Trapezoidal integration
                            area = 0.5 * (prev_val + curr_val) * (curr_X - prev_X)

                            if area_used + area > reactor['volume']:
                                # Interpolate to find exact X
                                remaining_vol = reactor['volume'] - area_used
                                slope = (curr_val - prev_val) / (curr_X - prev_X)

                                # Solve quadratic equation for dx
                                a = 0.5 * slope
                                b = prev_val
                                c = -remaining_vol

                                dx = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                                current_X = prev_X + dx
                                area_used = reactor['volume']
                                break

                            area_used += area
                            prev_X = curr_X
                            prev_val = curr_val

                        else:
                            current_X = subset.iloc[-1]['X']

                        cumulative_area += area_used

                # Step 7: Display results
                st.success(f"Maximum achievable conversion: {current_X:.4f}")
                st.info(f"Total volume used: {cumulative_area:.2f} L out of {total_volume:.2f} L")

                # Visualize the reactors on the plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(dataframe['X'], dataframe['F_A0/-r_A'], 'b-', linewidth=2, label='Levenspiel Curve')

                # Plot each reactor's contribution
                current_X_plot = 0.0
                colors = ['g', 'r', 'c', 'm', 'y']

                for i, reactor in enumerate(reactors):
                    if reactor['type'] == "CSTR":
                        # Create a fine grid for plotting
                        plot_X = np.linspace(current_X_plot, dataframe['X'].max(), 1000)
                        plot_rA = np.interp(plot_X, dataframe['X'], dataframe['r_A'])
                        plot_volumes = F_A0 * (plot_X - current_X_plot) / (-plot_rA)

                        valid_indices = np.where(plot_volumes <= reactor['volume'])[0]
                        if len(valid_indices) == 0:
                            break

                        max_index = valid_indices[-1]
                        new_X_plot = plot_X[max_index]
                        height = F_A0 / (-np.interp(new_X_plot, dataframe['X'], dataframe['r_A']))

                        ax2.plot([current_X_plot, new_X_plot], [height, height],
                                 f'{colors[i % len(colors)]}--', linewidth=2,
                                 label=f'CSTR {i + 1} (V={reactor["volume"]}L)')
                        ax2.plot([new_X_plot, new_X_plot], [0, height],
                                 f'{colors[i % len(colors)]}:')

                        current_X_plot = new_X_plot

                    elif reactor['type'] == "PFR/PBR":
                        # Plot the area under curve for PFR
                        subset = dataframe[(dataframe['X'] >= current_X_plot)]
                        if len(subset) < 2:
                            break

                        area_used = 0
                        prev_X = current_X_plot
                        prev_val = F_A0 / (-subset.iloc[0]['r_A'])
                        x_points = [prev_X]
                        y_points = [prev_val]

                        for idx in range(1, len(subset)):
                            curr_X = subset.iloc[idx]['X']
                            curr_val = F_A0 / (-subset.iloc[idx]['r_A'])

                            area = 0.5 * (prev_val + curr_val) * (curr_X - prev_X)

                            if area_used + area > reactor['volume']:
                                remaining_vol = reactor['volume'] - area_used
                                slope = (curr_val - prev_val) / (curr_X - prev_X)

                                a = 0.5 * slope
                                b = prev_val
                                c = -remaining_vol

                                dx = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                                final_X = prev_X + dx
                                final_val = prev_val + slope * dx

                                x_points.append(final_X)
                                y_points.append(final_val)
                                break

                            area_used += area
                            x_points.append(curr_X)
                            y_points.append(curr_val)
                            prev_X = curr_X
                            prev_val = curr_val

                        ax2.fill_between(x_points, y_points, color=colors[i % len(colors)],
                                         alpha=0.3, label=f'PFR {i + 1} (V={reactor["volume"]}L)')
                        current_X_plot = x_points[-1]

                ax2.set_xlabel('Conversion (X)')
                ax2.set_ylabel('F_A0 / -r_A (L)')
                ax2.set_title('Reactor Sequence Visualization')
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

