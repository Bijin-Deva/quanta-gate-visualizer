import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import plotly.graph_objects as go
from qiskit.quantum_info import DensityMatrix, partial_trace
import matplotlib.pyplot as plt
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError
)


# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantum Circuit Simulator")

# --- Gate Definitions ---
GATE_DEFINITIONS = {
    'I': {'name': 'Identity', 'color': '#6c757d'},
    'H': {'name': 'Hadamard', 'color': '#0d6efd'},
    'X': {'name': 'Pauli-X', 'color': '#dc3545'},
    'Y': {'name': 'Pauli-Y', 'color': '#dc3545'},
    'Z': {'name': 'Pauli-Z', 'color': '#dc3545'},
    'S': {'name': 'S Gate', 'color': '#ffc107'},
    'T': {'name': 'T Gate', 'color': '#ffc107'},
    '●': {'name': 'Control', 'color': '#198754'},
    '⊕': 'Target (X)', # Special case for display
}

# --- Helper Functions & State Management ---

def initialize_state(num_qubits, num_steps):
    """Initializes or resets the circuit grid and active gate."""
    st.session_state.circuit_grid = [['I'] * num_steps for _ in range(num_qubits)]
    if 'active_gate' not in st.session_state:
        st.session_state.active_gate = 'H'

def set_active_gate(gate_symbol):
    """Callback to set the currently selected gate."""
    st.session_state.active_gate = gate_symbol

def place_gate(q, t):
    """Callback to place the active gate on the grid."""
    active = st.session_state.active_gate
    if active == 'CNOT':
        st.session_state.circuit_grid[q][t] = '●'
        st.session_state.active_gate = '⊕'
    elif active == '⊕':
        st.session_state.circuit_grid[q][t] = '⊕'
        st.session_state.active_gate = 'H'
    else:
        st.session_state.circuit_grid[q][t] = active

def create_interactive_bloch_sphere(bloch_vector, title=""):
    x, y, z = bloch_vector

    fig = go.Figure()

    # --- Sphere surface ---
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.15,
        colorscale='Greys',
        showscale=False
    ))

    # --- Grid lines (meridians) ---
    for angle in range(0, 360, 30):
        a = np.deg2rad(angle)
        fig.add_trace(go.Scatter3d(
            x=np.cos(a)*np.sin(v),
            y=np.sin(a)*np.sin(v),
            z=np.cos(v),
            mode='lines',
            line=dict(color='lightgrey', width=1),
            showlegend=False
        ))

    # --- Grid lines (parallels) ---
    for angle in range(0, 180, 30):
        a = np.deg2rad(angle)
        fig.add_trace(go.Scatter3d(
            x=np.cos(u)*np.sin(a),
            y=np.sin(u)*np.sin(a),
            z=np.cos(a)*np.ones_like(u),
            mode='lines',
            line=dict(color='lightgrey', width=1),
            showlegend=False
        ))

    # --- Axes ---
    axis_len = 1.2
    axis_color = 'black'

    fig.add_trace(go.Scatter3d(x=[-axis_len, axis_len], y=[0,0], z=[0,0],
                               mode='lines', line=dict(color=axis_color, width=2)))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[-axis_len, axis_len], z=[0,0],
                               mode='lines', line=dict(color=axis_color, width=2)))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-axis_len, axis_len],
                               mode='lines', line=dict(color=axis_color, width=2)))

    # --- Axis labels ---
    fig.add_trace(go.Scatter3d(x=[1.3], y=[0], z=[0], mode='text', text=['X']))
    fig.add_trace(go.Scatter3d(x=[0], y=[1.3], z=[0], mode='text', text=['Y']))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1.3], mode='text', text=['|0⟩']))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-1.3], mode='text', text=['|1⟩']))

    # --- Bloch vector (thick line) ---
    r = np.sqrt(x*x + y*y + z*z)
    if r > 0.01:
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color='#FF1493', width=8),
            showlegend=False
        ))

        # --- Arrow head ---
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[x/r], v=[y/r], w=[z/r],
            anchor="tip",
            sizemode="absolute",
            sizeref=0.2,
            colorscale=[[0, '#FF1493'], [1, '#FF1493']],
            showscale=False
        ))

        # --- Tip marker ---
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=6, color='#FF1493'),
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def build_noise_model():
    noise = NoiseModel()

    if depol_p > 0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(depol_p, 1),
            ['h', 'x', 'y', 'z', 's', 't']
        )

    if decay_f > 0:
        noise.add_all_qubit_quantum_error(
            amplitude_damping_error(decay_f),
            ['h', 'x', 'y', 'z', 's', 't']
        )

    if phase_g > 0:
        noise.add_all_qubit_quantum_error(
            phase_damping_error(phase_g),
            ['h', 'x', 'y', 'z', 's', 't']
        )

    if tsp_01 > 0 or tsp_10 > 0:
        noise.add_all_qubit_readout_error(
            ReadoutError([
                [1 - tsp_01, tsp_01],
                [tsp_10, 1 - tsp_10]
            ])
        )

    return noise


# --- Streamlit UI ---
st.title('⚛️ Quantum Circuit Simulator')
st.markdown("Select a gate from the sidebar, then click on the grid to place it.")

# --- Sidebar ---
with st.sidebar:
    st.header('Circuit Controls')
    num_qubits = st.slider('Number of Qubits', 1, 5, 2, key='num_qubits_slider')
    num_steps = st.slider('Circuit Depth', 5, 15, 10, key='num_steps_slider')
    num_shots = st.slider('Number of Shots (for measurement)', 100, 4000, 1024, key='shots_slider')
    st.header("Quantum Noise")
    enable_noise = st.checkbox("Enable Noise", value=False)

    with st.expander("Noise Parameters"):
        depol_p = st.slider("Depolarization", 0.0, 0.3, 0.0)
        decay_f = st.slider("Amplitude Damping (T1)", 0.0, 0.3, 0.0)
        phase_g = st.slider("Phase Damping (T2)", 0.0, 0.3, 0.0)
        tsp_01 = st.slider("|0⟩ → |1⟩ (Readout)", 0.0, 0.3, 0.0)
        tsp_10 = st.slider("|1⟩ → |0⟩ (Readout)", 0.0, 0.3, 0.0)

    
    if 'circuit_grid' not in st.session_state or len(st.session_state.circuit_grid) != num_qubits or len(st.session_state.circuit_grid[0]) != num_steps:
        initialize_state(num_qubits, num_steps)

    if st.button('Reset Circuit', use_container_width=True):
        initialize_state(num_qubits, num_steps)
        st.success("Circuit reset!")

    st.header("Gate Palette")
    st.write("Current Gate: **" + st.session_state.active_gate + "**")
    
    gate_palette_cols = st.columns(2)
    palette_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'CNOT']
    for i, gate in enumerate(palette_gates):
        gate_palette_cols[i % 2].button(
            gate, on_click=set_active_gate, args=(gate,), use_container_width=True
        )
    if st.session_state.active_gate == '⊕':
        st.info("Now, click a grid cell to place the CNOT Target (⊕).")

# --- Main Circuit Grid UI ---
st.header('Quantum Circuit')
grid_cols = st.columns(num_steps + 1)
grid_cols[0].markdown("---") 

for i in range(num_steps):
    grid_cols[i + 1].markdown(f"<p style='text-align: center;'>{i}</p>", unsafe_allow_html=True)

for q in range(num_qubits):
    grid_cols[0].markdown(f"`|q{q}⟩`")
    for t in range(num_steps):
        gate_in_cell = st.session_state.circuit_grid[q][t]
        grid_cols[t + 1].button(
            gate_in_cell, key=f"cell_{q}_{t}", on_click=place_gate, args=(q, t), use_container_width=True
        )

# --- Execution Logic ---
if st.button('▶️ Execute', type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating circuit..."):
            # --- Build the Circuit from the Grid ---
            qc = QuantumCircuit(num_qubits)
            for t in range(num_steps):
                control_qubit = -1
                target_qubit = -1
                # First pass to find CNOTs in the current time step
                for q in range(num_qubits):
                    gate = st.session_state.circuit_grid[q][t]
                    if gate == '●':
                        control_qubit = q
                    elif gate == '⊕':
                        target_qubit = q
                
                # Apply gates for the current time step
                if control_qubit != -1 and target_qubit != -1:
                    qc.cx(control_qubit, target_qubit)
                elif control_qubit != -1 or target_qubit != -1:
                    # If only one part of CNOT is present, raise an error
                    raise ValueError(f"Incomplete CNOT gate in time step {t}.")
                else:
                    # Apply single-qubit gates if no CNOT in this step
                    for q in range(num_qubits):
                        gate = st.session_state.circuit_grid[q][t]
                        if gate != 'I' and gate != '●' and gate != '⊕':
                            getattr(qc, gate.lower())(q)
            
            st.success("✅ Simulation complete!")

            # --- Circuit Visualization ---
            st.header("Circuit Diagram")
            fig, ax = plt.subplots()
            qc.draw('mpl', ax=ax, style='iqx')
            st.pyplot(fig)
            plt.close(fig)
            
            # --- Measurement Simulation & Histogram ---
            st.header("Measurement Outcomes")
            qc_measured = qc.copy()
            qc_measured.measure_all()
            
            qasm_backend = Aer.get_backend('qasm_simulator')

            noise_model = build_noise_model() if enable_noise else None

            qasm_job = qasm_backend.run(
            qc_measured,
            shots=num_shots,
            noise_model=noise_model
            )

            counts = qasm_job.result().get_counts()
            
            if counts:
                # Find the outcome with the highest count
                most_likely_outcome = max(counts, key=counts.get)
                st.metric(label="Most Probable Classical Outcome", value=most_likely_outcome)

                # --- NEW CODE ADDED HERE ---
                qubit_order_str = "".join([f"q{i}" for i in range(num_qubits - 1, -1, -1)])
                st.info(f"💡 **How to Read the Output:** The bit string is ordered from highest to lowest qubit index ({qubit_order_str}). Qubit q0 is the rightmost digit.")
                # --- END OF NEW CODE ---

                sorted_counts = dict(sorted(counts.items()))
                hist_fig = go.Figure(go.Bar(
                    x=list(sorted_counts.keys()), 
                    y=list(sorted_counts.values()),
                    marker_color='indianred'
                ))
                hist_fig.update_layout(
                    title=f"Results from {num_shots} shots",
                    xaxis_title="Outcome (Classical Bit String)",
                    yaxis_title="Counts",
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                # --- Raw Counts Display ---
                st.subheader("Raw Measurement Counts")
                with st.expander("Show raw counts for each outcome"):
                    st.json(sorted_counts)

            else:
                st.warning("No measurement outcomes were recorded.")

            if enable_noise:
                st.info(
                "ℹ️ Bloch spheres show the ideal (noise-free) quantum state. "
                "Noise affects measurement outcomes and purity indirectly."
                )


            # --- Ideal State Simulation & Per-Qubit Results ---
            st.header("Ideal State Analysis (per Qubit)")
            st.markdown("This shows the theoretical quantum state of each qubit *before* measurement.")
            
            statevector_backend = Aer.get_backend('statevector_simulator')
            job = statevector_backend.run(qc)
            final_state = job.result().get_statevector()
            final_dm = DensityMatrix(final_state)

            # --- Display Per-Qubit Information ---
            cols = st.columns(num_qubits)
            for i in range(num_qubits):
                # Isolate the density matrix for the current qubit
                q_list = list(range(num_qubits))
                q_list.remove(i)
                reduced_dm = partial_trace(final_dm, q_list)
                
                # Calculate Bloch vector components
                x = np.real(np.trace(reduced_dm.data @ np.array([[0, 1], [1, 0]])))
                y = np.real(np.trace(reduced_dm.data @ np.array([[0, -1j], [1j, 0]])))
                z = np.real(np.trace(reduced_dm.data @ np.array([[1, 0], [0, -1]])))
                bloch_vector = [x, y, z]

                # Calculate probabilities from the diagonal of the reduced density matrix
                prob_0 = np.real(reduced_dm.data[0, 0])
                prob_1 = np.real(reduced_dm.data[1, 1])

                # Calculate purity
                purity = np.real(np.trace(reduced_dm.data @ reduced_dm.data))

                with cols[i]:
                    st.subheader(f"Qubit {i}")

                    # Display Bloch Sphere first
                    fig = create_interactive_bloch_sphere(bloch_vector)
                    st.plotly_chart(fig, use_container_width=True, key=f"bloch_sphere_{i}")

                    # Display analysis below the sphere
                    st.text(f"|0⟩: {prob_0:.3f}")
                    st.progress(prob_0)
                    st.text(f"|1⟩: {prob_1:.3f}")
                    st.progress(prob_1)
                    
                    st.metric(label="Purity", value=f"{purity:.3f}")

                    with st.expander("Details"):
                        st.text(f"Bloch Vector: ({x:.3f}, {y:.3f}, {z:.3f})")
                        st.text("Reduced Density Matrix:")
                        # Use st.dataframe to display the matrix cleanly
                        st.dataframe(reduced_dm.data)

    except ValueError as e:
        st.error(f"Circuit Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")




